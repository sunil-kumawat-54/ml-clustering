# -----------------------------------------------------------------------------
# 🔬 ClusterLab — Interactive ML Clustering Explorer
# -----------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import data_loader

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="ClusterLab", page_icon="🔬")

# -----------------------------------------------------------------------------
# 2. SESSION STATE MANAGEMENT
# -----------------------------------------------------------------------------
if "selected_algo" not in st.session_state:
    st.session_state.selected_algo = "🏠 Home / Overview"
if "selected_dataset" not in st.session_state:
    st.session_state.selected_dataset = "Blobs (Easy)"

# -----------------------------------------------------------------------------
# 2.5 GLOBAL CACHED HELPERS
# -----------------------------------------------------------------------------
@st.cache_data
def run_all_algorithms(X, k, eps, min_samples):
    import time
    results = {}
    
    # 1. K-Means
    t0 = time.perf_counter()
    km = KMeans(n_clusters=k, init="k-means++", random_state=42)
    km_labels = km.fit_predict(X)
    km_time = (time.perf_counter() - t0) * 1000.0
    results["K-Means"] = {
        "labels": km_labels,
        "time": km_time,
        "n_clusters": k,
        "error": False
    }
    
    # 2. DBSCAN
    t0 = time.perf_counter()
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = db.fit_predict(X)
    db_time = (time.perf_counter() - t0) * 1000.0
    n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    results["DBSCAN"] = {
        "labels": db_labels,
        "time": db_time,
        "n_clusters": n_db_clusters,
        "error": False
    }
    
    # 3. Hierarchical
    t0 = time.perf_counter()
    try:
        hr = AgglomerativeClustering(n_clusters=k, linkage='ward')
        hr_labels = hr.fit_predict(X)
        hr_time = (time.perf_counter() - t0) * 1000.0
        results["Hierarchical (Agglomerative)"] = {
            "labels": hr_labels,
            "time": hr_time,
            "n_clusters": k,
            "error": False
        }
    except Exception:
        results["Hierarchical (Agglomerative)"] = {
            "labels": None,
            "time": None,
            "n_clusters": 0,
            "error": True
        }
        
    # 4. GMM
    t0 = time.perf_counter()
    try:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        gmm_labels = gmm.fit_predict(X)
        gmm_time = (time.perf_counter() - t0) * 1000.0
        results["Gaussian Mixture Model"] = {
            "labels": gmm_labels,
            "time": gmm_time,
            "n_clusters": k,
            "error": False
        }
    except Exception:
        results["Gaussian Mixture Model"] = {
            "labels": None,
            "time": None,
            "n_clusters": 0,
            "error": True
        }
        
    # Compute evaluation metrics
    for name, res in results.items():
        if res["error"] or res["labels"] is None:
            res["silhouette"] = "N/A"
            res["ch"] = "N/A"
            res["db"] = "N/A"
            continue
            
        labels = res["labels"]
        unique_total = len(set(labels))
        
        # Silhouette, CH, and DB scores are only valid when 2 <= unique_labels < len(X)
        if unique_total < 2 or unique_total >= len(X):
            res["silhouette"] = "N/A"
            res["ch"] = "N/A"
            res["db"] = "N/A"
        else:
            try:
                res["silhouette"] = float(silhouette_score(X, labels))
                res["ch"] = float(calinski_harabasz_score(X, labels))
                res["db"] = float(davies_bouldin_score(X, labels))
            except Exception:
                res["silhouette"] = "N/A"
                res["ch"] = "N/A"
                res["db"] = "N/A"
                
    return results

@st.cache_data
def compute_dbscan_heatmap(X, metric):
    eps_grid = np.linspace(0.1, 1.5, 15)
    min_pts_grid = list(range(2, 16))
    heatmap_data = np.zeros((len(min_pts_grid), len(eps_grid)))
    
    for i, min_pts in enumerate(min_pts_grid):
        for j, eps_val in enumerate(eps_grid):
            db = DBSCAN(eps=eps_val, min_samples=min_pts, metric=metric)
            labels_temp = db.fit_predict(X)
            n_clusters = len(set(labels_temp)) - (1 if -1 in labels_temp else 0)
            heatmap_data[i, j] = n_clusters
            
    return heatmap_data, eps_grid, min_pts_grid

@st.cache_data
def fit_hierarchical(X, n_clusters, linkage, metric):
    actual_metric = "euclidean" if linkage == "ward" else metric
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        metric=actual_metric
    )
    labels = model.fit_predict(X)
    return labels

@st.cache_data
def compute_linkage_matrix(X, linkage, metric):
    scipy_metric = "cityblock" if metric == "manhattan" else metric
    actual_metric = "euclidean" if linkage == "ward" else scipy_metric
    Z = sch.linkage(X, method=linkage, metric=actual_metric)
    return Z

def get_ellipse_coords(mean, cov, n_std=1.0, n_points=100):
    if np.isscalar(cov) or cov.ndim == 0:
        cov_matrix = np.array([[cov, 0.0], [0.0, cov]])
    elif cov.ndim == 1:
        cov_matrix = np.diag(cov)
    else:
        cov_matrix = cov
        
    vals, vecs = np.linalg.eigh(cov_matrix)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    
    theta = np.arctan2(vecs[1, 0], vecs[0, 0])
    width = 2.0 * n_std * np.sqrt(max(vals[0], 1e-8))
    height = 2.0 * n_std * np.sqrt(max(vals[1], 1e-8))
    
    t = np.linspace(0, 2*np.pi, n_points)
    xs = (width / 2.0) * np.cos(t)
    ys = (height / 2.0) * np.sin(t)
    
    x_rot = xs * np.cos(theta) - ys * np.sin(theta) + mean[0]
    y_rot = xs * np.sin(theta) + ys * np.cos(theta) + mean[1]
    
    return x_rot, y_rot

@st.cache_data
def fit_gmm(X, n_components, covariance_type, max_iter):
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iter,
        random_state=42
    )
    labels = gmm.fit_predict(X)
    means = gmm.means_
    covariances = gmm.covariances_
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    log_lik = gmm.score(X) * len(X)
    converged = gmm.converged_
    return labels, means, covariances, bic, aic, log_lik, converged

@st.cache_data
def compute_gmm_information_criteria(X, covariance_type, max_iter):
    ks = list(range(1, 11))
    bics = []
    aics = []
    for k in ks:
        gmm_temp = GaussianMixture(
            n_components=k, 
            covariance_type=covariance_type, 
            max_iter=max_iter, 
            random_state=42
        )
        gmm_temp.fit(X)
        bics.append(gmm_temp.bic(X))
        aics.append(gmm_temp.aic(X))
    return ks, bics, aics

# -----------------------------------------------------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <h2 style="margin:0; color: #1e3c72; font-family: 'Outfit', sans-serif; font-size: 1.8rem;">🔬 ClusterLab</h2>
    <p style="font-size: 0.85rem; color: #666; margin-top: 5px;">Interactive Clustering Sandbox</p>
</div>
<hr style="margin: 10px 0 20px 0; border: none; height: 1px; background: #ddd;" />
""", unsafe_allow_html=True)

# Selectbox: Choose Algorithm
algo_options = [
    "🏠 Home / Overview", 
    "📌 K-Means", 
    "🌐 DBSCAN", 
    "🌲 Hierarchical (Agglomerative)", 
    "🎲 Gaussian Mixture Model",
    "📊 Compare All Algorithms"
]
selected_algo = st.sidebar.selectbox(
    "Choose Algorithm", 
    options=algo_options,
    key="selected_algo"
)

# Selectbox: Select Dataset
dataset_options = [
    "Blobs (Easy)", 
    "Moons (Non-linear)", 
    "Circles (Nested)", 
    "Iris (Real-world)", 
    "Upload CSV", 
    "Enter Text"
]
selected_dataset = st.sidebar.selectbox(
    "Select Dataset", 
    options=dataset_options,
    key="selected_dataset"
)

# Sidebar section for Hyperparameters
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Hyperparameters")
if selected_algo == "📌 K-Means":
    n_clusters = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=4, step=1, key="km_n_clusters")
    km_init = st.sidebar.selectbox("Centroid Initialization", options=["k-means++", "random"], index=0, key="km_init")
    max_iter = st.sidebar.slider("Max Iterations", min_value=50, max_value=500, value=300, step=50, key="km_max_iter")
    n_init = st.sidebar.slider("Number of Initializations (n_init)", min_value=1, max_value=20, value=10, step=1, key="km_n_init")
elif selected_algo == "🌐 DBSCAN":
    eps_val = st.sidebar.slider(r"Epsilon ($\varepsilon$)", min_value=0.05, max_value=2.0, value=0.5, step=0.05, key="db_eps")
    min_samples = st.sidebar.slider("Minimum Samples (minPts)", min_value=2, max_value=20, value=5, step=1, key="db_min_samples")
    db_metric = st.sidebar.selectbox("Distance Metric", options=["euclidean", "manhattan", "cosine"], index=0, key="db_metric")
elif selected_algo == "🌲 Hierarchical (Agglomerative)":
    hr_n_clusters = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=10, value=3, step=1, key="hr_n_clusters")
    hr_linkage = st.sidebar.selectbox("Linkage Method", options=["ward", "complete", "average", "single"], index=0, key="hr_linkage")
    if hr_linkage == "ward":
        hr_metric = st.sidebar.selectbox("Distance Metric", options=["euclidean"], index=0, disabled=True, key="hr_metric", help="Ward linkage requires Euclidean distance.")
    else:
        hr_metric = st.sidebar.selectbox("Distance Metric", options=["euclidean", "manhattan", "cosine"], index=0, key="hr_metric")
elif selected_algo == "🎲 Gaussian Mixture Model":
    gmm_n_components = st.sidebar.slider("Number of Components (k)", min_value=2, max_value=10, value=3, step=1, key="gmm_n_components")
    gmm_covariance_type = st.sidebar.selectbox(
        "Covariance Type", 
        options=["full", "tied", "diag", "spherical"], 
        index=0, 
        key="gmm_covariance_type",
        help="• full: Each cluster has its own general covariance matrix (arbitrary shapes)\n"
             "• tied: All clusters share the same covariance matrix\n"
             "• diag: Axis-aligned elliptical clusters\n"
             "• spherical: Circular/spherical clusters"
    )
    gmm_max_iter = st.sidebar.slider("Max Iterations", min_value=50, max_value=500, value=100, step=50, key="gmm_max_iter")
elif selected_algo == "📊 Compare All Algorithms":
    comp_k = st.sidebar.slider("Common Clusters (k) [K-Means, Hierarchical, GMM]", min_value=2, max_value=10, value=3, step=1, key="comp_k")
    comp_eps = st.sidebar.slider(r"DBSCAN Epsilon ($\varepsilon$)", min_value=0.05, max_value=2.0, value=0.5, step=0.05, key="comp_eps")
    comp_min_samples = st.sidebar.slider("DBSCAN Min Samples (minPts)", min_value=2, max_value=20, value=5, step=1, key="comp_min_samples")
else:
    st.sidebar.info("Select a clustering algorithm to configure its specific hyperparameters.")


# -----------------------------------------------------------------------------
# 4. CUSTOM DESIGN & CSS THEME
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        color: #1e3c72;
    }
    
    /* Elegant containers and margins */
    .theory-box {
        background-color: #f8fafc;
        border-left: 5px solid #2563eb;
        padding: 20px;
        border-radius: 4px 8px 8px 4px;
        margin-bottom: 20px;
    }
    
    .formula-box {
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid #e2e8f0;
    }
    
    .example-box {
        background-color: #fafafa;
        border-radius: 8px;
        padding: 20px;
        border: 1px dashed #cccccc;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Main Area Header Banner
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    <h1 style="margin: 0; font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 2.4rem; color: white; letter-spacing: -0.5px;">🔬 ClusterLab — Interactive ML Clustering Explorer</h1>
    <p style="margin: 6px 0 0 0; font-family: 'Inter', sans-serif; font-size: 1.1rem; opacity: 0.9; font-weight: 300;">
        Explore K-Means, DBSCAN, Hierarchical & Gaussian Mixture clustering interactively
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. DATASET LOADING PIPELINE
# -----------------------------------------------------------------------------
# Call data_loader functions based on sidebar selection
X_scaled = None
df_raw = None
dataset_description = ""

if selected_dataset in ["Blobs (Easy)", "Moons (Non-linear)", "Circles (Nested)", "Iris (Real-world)"]:
    X_scaled, df_raw = data_loader.load_builtin_dataset(selected_dataset)
    if selected_dataset == "Blobs (Easy)":
        dataset_description = "Synthetic dataset with 4 distinct, spherical clusters. Ideal for centroid-based clustering."
    elif selected_dataset == "Moons (Non-linear)":
        dataset_description = "Synthetic dataset with 2 interleaving half-circles. Designed for testing non-linear clustering capability."
    elif selected_dataset == "Circles (Nested)":
        dataset_description = "Synthetic dataset with nested concentric circles. Tests handling of structural spatial topology."
    elif selected_dataset == "Iris (Real-world)":
        dataset_description = "Classic real-world dataset containing 3 species of Iris flowers (150 samples, 4 numerical dimensions)."
elif selected_dataset == "Upload CSV":
    dataset_description = "Allows you to upload your own custom tabular dataset (.csv) for tailored clustering analysis."
    X_scaled, df_raw = data_loader.load_csv_dataset()
elif selected_dataset == "Enter Text":
    dataset_description = "Allows entering manual 2D coordinates for rapid exploratory data input."
    X_scaled, df_raw = data_loader.load_text_dataset()

# Store loaded data in st.session_state["X"] for use by all algorithm pages
if X_scaled is not None:
    st.session_state["X"] = X_scaled
    st.session_state["df_raw"] = df_raw
else:
    st.session_state["X"] = None
    st.session_state["df_raw"] = None


# -----------------------------------------------------------------------------
# 6. RENDER PAGE VIEWS
# -----------------------------------------------------------------------------

# --- A. HOME PAGE ---
if selected_algo == "🏠 Home / Overview":
    st.markdown("### Welcome to ClusterLab! 👋")
    st.write(
        "ClusterLab is a comprehensive interactive simulator for exploring unsupervised machine learning. "
        "Clustering is the process of partitioning data into groups so that elements within a group are "
        "more similar to each other than to those in other groups. Navigate using the sidebar to explore "
        "specific algorithms in depth."
    )
    
    # How to Use Guide
    st.markdown("#### 🚀 How to use this app")
    st.markdown("""
    - **Choose an Algorithm:** Use the **Sidebar dropdown** to select one of the four clustering models (K-Means, DBSCAN, Hierarchical, or Gaussian Mixture).
    - **Pick a Dataset:** Select a spatial dataset distribution (e.g., Blobs, Moons, or Nested Circles) or prep a custom CSV.
    - **Tweak Parameters & Explore:** Adjust hyperparameters in the sidebar (coming in Phase 2) and investigate algorithm properties under the **Theory, Formula, Live Demo**, and **Example** tabs!
    """)
    
    # Comparison Table
    st.markdown("#### 📊 Clustering Algorithms Comparison")
    comparison_data = {
        "Algorithm": ["📌 K-Means", "🌐 DBSCAN", "🌲 Hierarchical (Agglomerative)", "🎲 Gaussian Mixture Model"],
        "Type": ["Centroid-based", "Density-based", "Connectivity-based", "Distribution-based"],
        "Key Parameter": ["Number of clusters (k)", "Epsilon (ε) & Min Samples", "Linkage criteria & Distance metric", "Number of components (k) & Covariance type"],
        "Best For": [
            "Spherical, well-separated clusters; fast and scalable to large datasets.",
            "Arbitrary-shaped clusters; robustly detecting noise/outliers.",
            "Visualizing hierarchical relationships (dendrograms); smaller datasets.",
            "Ellipsoidal clusters; soft clustering (overlapping groups with probabilities)."
        ],
        "Weakness": [
            "Sensitive to outliers; assumes spherical clusters of similar size; requires pre-specifying k.",
            "Struggles with clusters of varying densities and high-dimensional data; sensitive to ε.",
            "High computational cost O(N³) and memory complexity; cannot undo step-wise merges.",
            "Highly sensitive to initialization; can converge to local minima or suffer from overfitting."
        ]
    }
    df_compare = pd.DataFrame(comparison_data)
    st.table(df_compare)
    
    # Live Dataset Preview on Home
    st.markdown("---")
    st.markdown("### 📊 Raw Data Preview")
    
    if st.session_state["X"] is not None:
        df_preview = st.session_state["df_raw"]
        features = [c for c in df_preview.columns if c not in ["Target", "Species"]]
        n = st.session_state["X"].shape[0]
        d = st.session_state["X"].shape[1]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"**Selected Dataset:** `{selected_dataset}`")
            st.info(dataset_description)
            st.metric("Data Points", n)
            st.metric("Features", d)
            st.markdown("**Feature List:**")
            for feat in features:
                st.write(f"- `{feat}`")
        
        with col2:
            fig = px.scatter(
                df_preview, 
                x=features[0], 
                y=features[1], 
                color_discrete_sequence=["#888888"],  # Neutral gray color
                title=f"Raw Data Distribution (Unclustered — {selected_dataset})"
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)',
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        if selected_dataset == "Upload CSV":
            st.warning("⚠️ No CSV has been uploaded yet. Please upload a valid CSV file using the file uploader above.")
        elif selected_dataset == "Enter Text":
            st.warning("⚠️ No valid text coordinates have been loaded yet. Please adjust the coordinates in the text box above.")



# --- B. K-MEANS PAGE ---
elif selected_algo == "📌 K-Means":
    st.subheader("📌 K-Means Clustering")
    tab_theory, tab_formula, tab_demo, tab_example = st.tabs(["📖 Theory", "🔢 Formula", "⚡ Live Demo", "💡 Example"])
    
    with tab_theory:
        st.markdown("""
        <div class="theory-box">
            <h4>Centroid-Based Hard Partitioning</h4>
            <p><strong>K-Means</strong> is an unsupervised learning algorithm that divides a dataset of $n$ samples into $k$ 
            distinct, non-overlapping clusters. Each point is assigned to a cluster represented by its mathematical mean, 
            known as a <strong>centroid</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_algo, col_assump = st.columns(2)
        
        with col_algo:
            st.markdown("##### 🔄 The 4-Step Algorithmic Loop:")
            st.markdown("""
            1. **Centroid Initialization:** Select $k$ initial centroids. By default, `k-means++` is used to smartly space them out, preventing poor convergence.
            2. **Cluster Assignment:** Assign each data point to its closest centroid using the Euclidean distance metric.
            3. **Centroid Update:** Recalculate each centroid's coordinates as the average (mean) of all data points belonging to its cluster.
            4. **Repeat Convergence Loop:** Iterate between the Assignment and Update steps until centroids stabilize (stop shifting) or maximum iterations are exceeded.
            """)
            
            st.info("""
            ⏰ **Algorithmic Time Complexity:**
            The computational complexity is **$O(n \cdot k \cdot i \cdot d)$**, where:
            - **$n$**: Number of data points.
            - **$k$**: Number of clusters.
            - **$i$**: Number of iterations until convergence.
            - **$d$**: Number of features (dimensions).
            
            This linear scaling makes K-Means extremely fast and scalable to massive datasets compared to hierarchical clustering.
            """)
            
        with col_assump:
            st.markdown("##### 📌 Key Geometric Assumptions:")
            st.markdown("""
            - **Convexity:** Assumes clusters are spherical (well-separated, round groupings). It cannot identify non-linear topologies (e.g. spirals or nested circles).
            - **Equal Variance:** Assumes clusters have similar densities and isotropic distributions.
            - **Equal Cluster Sizes:** Struggles when one cluster contains 1,000 points and another contains only 10 points.
            """)
            
            st.success("""
            ✔️ **When to use K-Means:**
            - For quick, baseline exploratory data analysis.
            - When clusters are well-separated, spherical, and of similar size.
            - When dealing with large volumes of data.
            """)
            
            st.warning("""
            ❌ **When NOT to use K-Means:**
            - When clusters contain complex, non-linear shapes (use DBSCAN or Spectral Clustering).
            - When the data has heavy outliers (outliers pull centroids and ruin boundaries).
            - When clusters have heavily varying densities.
            """)

        
    with tab_formula:
        st.markdown("#### 🔢 Mathematical Mechanics of K-Means")
        st.write(
            "K-Means works by optimizing centroid coordinates to minimize cluster dispersion. "
            "Below are the core equations driving this optimization:"
        )
        
        st.markdown("##### 1. Objective Function (Within-Cluster Sum of Squares - WCSS)")
        st.latex(r"J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \| x_i - \mu_j \|^2")
        st.caption("📖 **Plain English:** Minimizes the total squared distance between each data point and its assigned cluster centroid (also known as *inertia*).")
        
        st.markdown("##### 2. Centroid Update Equation")
        st.latex(r"\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i")
        st.caption("📖 **Plain English:** Repositions the centroid coordinate to the average (mean) value of all points currently assigned to that cluster.")
        
        st.markdown("##### 3. Euclidean Distance Metric")
        st.latex(r"d(x_i, \mu_j) = \sqrt{\sum_{l=1}^{d} (x_{il} - \mu_{jl})^2}")
        st.caption("📖 **Plain English:** Calculates the standard straight-line distance in $d$-dimensional space to assign points to their nearest centroid.")

        
    with tab_demo:
        st.markdown("#### ⚡ Live K-Means Sandbox")
        
        if st.session_state["X"] is not None:
            X = st.session_state["X"]
            df_raw = st.session_state["df_raw"]
            
            # 1. Fit KMeans Model
            kmeans = KMeans(
                n_clusters=n_clusters, 
                init=km_init, 
                max_iter=max_iter, 
                n_init=n_init, 
                random_state=42
            )
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_
            inertia = kmeans.inertia_
            n_iter = kmeans.n_iter_
            
            # Calculate Silhouette Score (requires at least 2 clusters and N > K)
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X, labels)
            else:
                sil = 0.0
                
            # 2. Main Layout - 2 Columns
            col_plot, col_metrics = st.columns([2, 1])
            
            with col_plot:
                # Build Clustered DataFrame
                df_clustered = pd.DataFrame(X, columns=["Feature 1 (Scaled)", "Feature 2 (Scaled)"])
                df_clustered["Cluster"] = [f"Cluster {l+1}" for l in labels]
                
                # Plotly Scatter Plot
                fig = px.scatter(
                    df_clustered, 
                    x="Feature 1 (Scaled)", 
                    y="Feature 2 (Scaled)", 
                    color="Cluster",
                    color_discrete_sequence=px.colors.qualitative.Dark2,
                    title=f"K-Means Clustering Results (k={n_clusters})"
                )
                
                # Add Centroids
                for idx, cent in enumerate(centroids):
                    fig.add_trace(go.Scatter(
                        x=[cent[0]],
                        y=[cent[1]],
                        mode='markers',
                        marker=dict(
                            symbol='star',
                            size=18,
                            color='yellow',
                            line=dict(color='black', width=2)
                        ),
                        name=f"Centroid {idx + 1}",
                        showlegend=True
                    ))
                    
                fig.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_metrics:
                st.markdown("##### 📈 Model Metrics")
                
                st.metric(
                    label="Inertia (Within-Cluster WCSS)", 
                    value=f"{inertia:.4f}",
                    help="Sum of squared distances of samples to their closest cluster center. Lower is better."
                )
                st.metric(
                    label="Silhouette Score", 
                    value=f"{sil:.4f}",
                    help="Measures how similar an object is to its own cluster compared to other clusters. Ranges from -1 to 1; higher is better."
                )
                st.metric(
                    label="Iterations to Converge", 
                    value=int(n_iter),
                    help="Number of iterations run until centroid positions stabilized."
                )
                
                st.info(
                    "💡 **Quick Tip:** Tweak hyperparameters in the sidebar (like $k$ or Initialization) "
                    "to see how the cluster boundaries and metrics respond instantly!"
                )
            
            # 3. Double Diagnostics Row
            st.markdown("---")
            st.markdown("### 📊 Clustering Hyperparameter Diagnostics")
            col_elbow, col_sil = st.columns(2)
            
            with col_elbow:
                # Elbow Chart
                ks = list(range(1, 11))
                inertias = []
                for k in ks:
                    km_temp = KMeans(n_clusters=k, init=km_init, max_iter=max_iter, n_init=n_init, random_state=42)
                    km_temp.fit(X)
                    inertias.append(km_temp.inertia_)
                
                df_elbow = pd.DataFrame({"Number of Clusters (k)": ks, "WCSS (Inertia)": inertias})
                fig_elbow = px.line(
                    df_elbow, 
                    x="Number of Clusters (k)", 
                    y="WCSS (Inertia)", 
                    markers=True, 
                    title="Elbow Method: WCSS vs. k"
                )
                fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="#E02424", annotation_text=f"Active k={n_clusters}")
                fig_elbow.update_layout(
                    margin=dict(l=20, r=20, t=45, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickmode='linear', tick0=1, dtick=1),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
                )
                st.plotly_chart(fig_elbow, use_container_width=True)
                
            with col_sil:
                # Silhouette vs. K
                ks_sil = list(range(2, 11))
                sils = []
                for k in ks_sil:
                    km_temp = KMeans(n_clusters=k, init=km_init, max_iter=max_iter, n_init=n_init, random_state=42)
                    km_temp.fit(X)
                    sils.append(silhouette_score(X, km_temp.labels_))
                    
                df_sil = pd.DataFrame({"Number of Clusters (k)": ks_sil, "Silhouette Score": sils})
                fig_sil = px.bar(
                    df_sil, 
                    x="Number of Clusters (k)", 
                    y="Silhouette Score", 
                    title="Silhouette Score vs. k"
                )
                fig_sil.add_vline(x=n_clusters, line_dash="dash", line_color="#E02424", annotation_text=f"Active k={n_clusters}")
                
                # Highlight active k bar
                colors = ["#4B5563"] * len(ks_sil)
                if n_clusters in ks_sil:
                    colors[ks_sil.index(n_clusters)] = "#3B82F6"
                fig_sil.update_traces(marker_color=colors)
                
                fig_sil.update_layout(
                    margin=dict(l=20, r=20, t=45, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickmode='linear', tick0=2, dtick=1),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
                )
                st.plotly_chart(fig_sil, use_container_width=True)
        else:
            if selected_dataset == "Upload CSV":
                st.warning("⚠️ Please upload a valid CSV file first to activate the live clustering demo.")
            elif selected_dataset == "Enter Text":
                st.warning("⚠️ Please provide valid numeric coordinates first to activate the live clustering demo.")

            
    with tab_example:
        st.markdown("#### 💡 Worked Example: Iris Sepal Segmentation")
        
        # Real-world analogy
        st.success("""
        🪙 **Real-World Analogy:** 
        *K-Means is like sorting coins by size.* Imagine having a pile of mixed coins of various diameters, and you want to 
        sort them into exactly 3 groups. You start by making 3 random guesses of coin sizes (Centroids). Then, you compare each 
        coin in your pile and place it into the group of the guess it is closest to (Assignment). Finally, you look at each of 
        the three piles and calculate their exact average size, shifting your guesses to those averages (Update). 
        You repeat this until the piles stop shifting. You have sorted the coins without ever knowing their names or values!
        """)
        
        st.write(
            "Below is a complete walkthrough of using K-Means to cluster the classic Iris dataset based on "
            "Sepal Length and Sepal Width in 2D space."
        )
        
        # Walkthrough Steps
        st.markdown("##### 🏃‍♂️ Step-by-Step Walkthrough:")
        
        st.markdown("###### **Step 1: Load and scale features**")
        st.write("We load the dataset using `scikit-learn` and apply `StandardScaler` to ensure features have equal variance.")
        st.code("""
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load Iris dataset
iris = load_iris()
X_raw = iris.data[:, :2] # Sepal Length, Sepal Width

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
        """, language="python")
        
        st.markdown("###### **Step 2: Fit K-Means with 3 clusters**")
        st.write("Since we know there are 3 distinct species, we configure `n_clusters=3`.")
        st.code("""
from sklearn.cluster import KMeans

# Instantiate and fit K-Means
kmeans_iris = KMeans(n_clusters=3, init="k-means++", random_state=42)
labels = kmeans_iris.fit_predict(X_scaled)
centroids = kmeans_iris.cluster_centers_
        """, language="python")
        
        st.markdown("###### **Step 3: Visualize Clustered Results**")
        st.write("Observe the final groupings plotted below with their converged cluster centroids:")
        
        # Live local run for example tab
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        
        iris = load_iris()
        X_iris_raw = iris.data[:, :2]
        scaler_iris = StandardScaler()
        X_iris_scaled = scaler_iris.fit_transform(X_iris_raw)
        
        kmeans_iris = KMeans(n_clusters=3, init="k-means++", random_state=42)
        labels_iris = kmeans_iris.fit_predict(X_iris_scaled)
        centroids_iris = kmeans_iris.cluster_centers_
        
        df_iris = pd.DataFrame(X_iris_scaled, columns=["Sepal Length (Scaled)", "Sepal Width (Scaled)"])
        df_iris["Species Group"] = [f"Species Group {l+1}" for l in labels_iris]
        
        fig_iris = px.scatter(
            df_iris, 
            x="Sepal Length (Scaled)", 
            y="Sepal Width (Scaled)", 
            color="Species Group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Iris Sepal Clusters (Standardized Space)"
        )
        
        for idx, cent in enumerate(centroids_iris):
            fig_iris.add_trace(go.Scatter(
                x=[cent[0]],
                y=[cent[1]],
                mode='markers',
                marker=dict(
                    symbol='star',
                    size=16,
                    color='yellow',
                    line=dict(color='black', width=2)
                ),
                name=f"Centroid {idx + 1}",
                showlegend=True
            ))
            
        fig_iris.update_layout(
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.05)',
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
        )
        st.plotly_chart(fig_iris, use_container_width=True)

# --- C. DBSCAN PAGE ---
elif selected_algo == "🌐 DBSCAN":
    st.subheader("🌐 DBSCAN Clustering")
    tab_theory, tab_formula, tab_demo, tab_example = st.tabs(["📖 Theory", "🔢 Formula", "⚡ Live Demo", "💡 Example"])
    
    with tab_theory:
        st.markdown("""
        <div class="theory-box">
            <h4>Density-Based spatial Clustering of Applications with Noise</h4>
            <p><strong>DBSCAN</strong> is a powerful unsupervised algorithm that discovers clusters based on spatial density. 
            Unlike K-Means, it does not require specifying the number of clusters $k$ in advance. Instead, it traces high-density 
            groupings and labels sparse regions as noise/outliers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_concept, col_diag = st.columns([1, 1])
        
        with col_concept:
            st.markdown("##### ⚙️ The Two Crucial Parameters:")
            st.markdown("""
            - **Epsilon ($\varepsilon$):** The maximum radius of the neighborhood to search around any given point.
            - **Minimum Samples ($\text{minPts}$):** The minimum number of points required within the $\varepsilon$-neighborhood to define a cluster core.
            """)
            
            st.markdown("##### 📍 The Three Types of Points:")
            st.markdown("""
            - 🔴 **Core Points:** Points that have at least $\text{minPts}$ neighbors (including themselves) within their $\varepsilon$-neighborhood.
            - 🟡 **Border Points:** Points that are within $\varepsilon$ of a core point but do not satisfy the minimum neighborhood density themselves.
            - ⚪ **Noise Points (Outliers):** All other points that are neither core nor border points (e.g. they fall in sparse countryside regions).
            """)
            
            st.success("""
            ✔️ **DBSCAN Strengths:**
            - Automatically discovers clusters of **arbitrary shapes** (crescents, rings, interlocking streams).
            - Natively handles noise and filters out outliers.
            - No need to guess or pre-specify cluster counts.
            """)
            
            st.warning("""
            ❌ **DBSCAN Limitations:**
            - Struggles when clusters have **heavily varying densities** (no single $\varepsilon$ works globally).
            - Extremely sensitive to the choice of $\varepsilon$ and $\text{minPts}$.
            - Suffers in high dimensions where distance vectors become sparse.
            """)
            
        with col_diag:
            st.markdown("##### 📊 Point Classification Illustration")
            
            # Construct a small static dataset to illustrate DBSCAN classification
            x_static = [0.0, 0.3, -0.5, 0.2, 0.85, 2.3]
            y_static = [0.0, -0.4, 0.4, 0.5, 0.6, 2.0]
            types_static = [
                "🔴 Core Point (0,0)", 
                "🔴 Core Neighbor", 
                "🔴 Core Neighbor", 
                "🔴 Core Neighbor", 
                "🟡 Border Point", 
                "⚪ Noise Point"
            ]
            
            df_static = pd.DataFrame({"x": x_static, "y": y_static, "Type": types_static})
            fig_static = px.scatter(
                df_static, x="x", y="y", color="Type",
                color_discrete_map={
                    "🔴 Core Point (0,0)": "#EF4444",
                    "🔴 Core Neighbor": "#FCA5A5",
                    "🟡 Border Point": "#FBBF24",
                    "⚪ Noise Point": "#9CA3AF"
                }
            )
            
            # Draw epsilon neighborhood circle around Core Point (0,0) with radius epsilon = 1.0
            fig_static.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=-1.0, y0=-1.0, x1=1.0, y1=1.0,
                line=dict(color="rgba(239, 68, 68, 0.5)", width=2, dash="dash"),
                fillcolor="rgba(239, 68, 68, 0.05)"
            )
            # Draw epsilon neighborhood circle around Border Point (0.85,0.6) with radius epsilon = 1.0
            fig_static.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=-0.15, y0=-0.4, x1=1.85, y1=1.6,
                line=dict(color="rgba(251, 191, 36, 0.5)", width=2, dash="dot"),
                fillcolor="rgba(251, 191, 36, 0.02)"
            )
            
            fig_static.update_layout(
                xaxis=dict(range=[-1.5, 3.2], showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=True),
                yaxis=dict(range=[-1.5, 3.2], showgrid=True, gridcolor='rgba(0,0,0,0.1)', zeroline=True),
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)',
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0.0)
            )
            st.plotly_chart(fig_static, use_container_width=True)

        
    with tab_formula:
        st.markdown("#### 🔢 Mathematical Mechanics of DBSCAN")
        st.write(
            "DBSCAN operates on the concepts of neighborhood densities and connectivity pathways. "
            "Below are the primary mathematical formulations driving DBSCAN's logic:"
        )
        
        st.markdown("##### 1. Epsilon-Neighborhood Definition")
        st.latex(r"N_{\varepsilon}(p) = \{ q \in D \mid \text{dist}(p, q) \le \varepsilon \}")
        st.caption("📖 **Plain English:** Defines the localized neighborhood sphere of radius Epsilon ($\varepsilon$) surrounding a point $p$.")
        
        st.markdown("##### 2. Core Density Condition")
        st.latex(r"|N_{\varepsilon}(p)| \ge \text{minPts}")
        st.caption("📖 **Plain English:** A point $p$ is designated as a Core Point if its local neighborhood contains at least $\text{minPts}$ (including $p$ itself).")
        
        st.markdown("##### 3. Euclidean Distance Metric")
        st.latex(r"\text{dist}(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}")
        st.caption("📖 **Plain English:** The straight-line Euclidean distance metric used globally in $n$-dimensional coordinate space to check neighborhood containment.")
        
        st.markdown("---")
        st.markdown("##### 🔄 Core Concepts of Reachability & Connectivity")
        
        col_reach, col_conn = st.columns(2)
        with col_reach:
            st.markdown("###### **Direct & Indirect Density-Reachability**")
            st.write(
                "A point $q$ is **directly density-reachable** from $p$ if $q$ is inside the $\varepsilon$-neighborhood of $p$ "
                "and $p$ is a core point. If we can traverse a chain of directly reachable core points from $p$ to $w$, "
                "then $w$ is **density-reachable** from $p$."
            )
        with col_conn:
            st.markdown("###### **Density-Connectivity**")
            st.write(
                "Two points $p$ and $q$ are **density-connected** if there exists an intermediate point $o$ such that "
                "both $p$ and $q$ are density-reachable from $o$. Density-connectivity is a symmetric relationship that allows DBSCAN "
                "to trace paths through neighboring core points and discover arbitrarily shaped clusters of any topology."
            )

        
    with tab_demo:
        st.markdown("#### ⚡ Live DBSCAN Sandbox")
        
        if st.session_state["X"] is not None:
            X = st.session_state["X"]
            df_raw = st.session_state["df_raw"]
            
            # 1. Fit DBSCAN Model
            dbscan = DBSCAN(eps=eps_val, min_samples=min_samples, metric=db_metric)
            labels = dbscan.fit_predict(X)
            unique_labels = np.unique(labels)
            
            # Check if everything is noise (all -1)
            if len(unique_labels) == 1 and unique_labels[0] == -1:
                st.error(
                    "⚠️ **All points classified as Noise!** The parameters ε (Epsilon) and minPts are too restrictive for this data. "
                    "Try **increasing Epsilon** or **decreasing Minimum Samples** in the sidebar to allow clusters to form."
                )
            else:
                n_clusters_found = len([l for l in unique_labels if l != -1])
                n_noise = list(labels).count(-1)
                n_core = len(dbscan.core_sample_indices_)
                
                # Calculate silhouette score (requires at least 2 clusters and valid labels)
                if len(set(labels)) > 1 and n_clusters_found > 1:
                    try:
                        sil = silhouette_score(X, labels)
                        sil_str = f"{sil:.4f}"
                    except Exception:
                        sil_str = "N/A"
                else:
                    sil_str = "N/A"
                    
                # 2. Main Layout - 2 Columns
                col_plot, col_metrics = st.columns([2, 1])
                
                with col_plot:
                    # Build Clustered DataFrame
                    df_clustered = pd.DataFrame(X, columns=["Feature 1 (Scaled)", "Feature 2 (Scaled)"])
                    df_clustered["Cluster"] = [f"Cluster {l}" if l != -1 else "Noise" for l in labels]
                    
                    fig = go.Figure()
                    
                    # Colors palette
                    colors_palette = px.colors.qualitative.Dark2
                    
                    # Separate clusters and noise to style them uniquely
                    for idx, l in enumerate(unique_labels):
                        if l == -1:
                            df_noise = df_clustered[df_clustered["Cluster"] == "Noise"]
                            fig.add_trace(go.Scatter(
                                x=df_noise["Feature 1 (Scaled)"],
                                y=df_noise["Feature 2 (Scaled)"],
                                mode='markers',
                                marker=dict(
                                    symbol='x',
                                    size=8,
                                    color='#9CA3AF'
                                ),
                                name="Noise / Outliers"
                            ))
                        else:
                            df_c = df_clustered[df_clustered["Cluster"] == f"Cluster {l}"]
                            color_hex = colors_palette[l % len(colors_palette)]
                            fig.add_trace(go.Scatter(
                                x=df_c["Feature 1 (Scaled)"],
                                y=df_c["Feature 2 (Scaled)"],
                                mode='markers',
                                marker=dict(
                                    symbol='circle',
                                    size=8,
                                    color=color_hex
                                ),
                                name=f"Cluster {l + 1}"
                            ))
                            
                    fig.update_layout(
                        title=f"DBSCAN Cluster Results (ε={eps_val:.2f}, minPts={min_samples})",
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.05)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col_metrics:
                    st.markdown("##### 📈 Model Metrics")
                    
                    st.metric(
                        label="Clusters Found", 
                        value=int(n_clusters_found),
                        help="Number of dense clusters discovered (excluding noise)."
                    )
                    st.metric(
                        label="Noise / Outliers", 
                        value=int(n_noise),
                        help="Number of points that do not fall in any dense cluster neighborhood."
                    )
                    st.metric(
                        label="Core Points", 
                        value=int(n_core),
                        help="Number of points that satisfy the local density condition."
                    )
                    st.metric(
                        label="Silhouette Score", 
                        value=sil_str,
                        help="Represents the separation distance between clusters. Shows 'N/A' if only 1 cluster was found."
                    )
                    
            # 3. Parameter Sensitivity Heatmap
            st.markdown("---")
            st.markdown("### 🗺️ Hyperparameter Sensitivity Heatmap")
            st.write(
                "The heatmap below precomputes the number of clusters discovered across a wide grid of Epsilon (ε) "
                "and Minimum Samples (minPts) configurations to help you find the most stable parameter plateau."
            )
            
            with st.spinner("Precomputing hyperparameter grid..."):
                heatmap, eps_g, pts_g = compute_dbscan_heatmap(X, db_metric)
                
            fig_heat = px.imshow(
                heatmap,
                x=[f"{e:.2f}" for e in eps_g],
                y=[str(p) for p in pts_g],
                labels=dict(x="Epsilon (ε)", y="Minimum Samples (minPts)", color="Clusters Found"),
                color_continuous_scale="Viridis",
                title="Parameter Grid Exploration (ε vs minPts)"
            )
            fig_heat.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            if selected_dataset == "Upload CSV":
                st.warning("⚠️ Please upload a valid CSV file first to activate the live clustering demo.")
            elif selected_dataset == "Enter Text":
                st.warning("⚠️ Please provide valid numeric coordinates first to activate the live clustering demo.")

            
    with tab_example:
        st.markdown("#### 💡 Worked Example: Non-Linear Moons Segmentation")
        
        # Real-world analogy
        st.success("""
        🗺️ **Real-World Analogy:** 
        *DBSCAN is like identifying cities on a map.* Imagine looking at a satellite map of lights at night. 
        Where lights are clustered together densely, you identify cities (Clusters). It doesn't matter if a city is round 
        like Paris or long and narrow like a coastal strip in Florida (Arbitrary shapes) — as long as there is continuous density, 
        it counts as the same city! In between cities, you have sparse lights from highways or isolated farmhouses. 
        You tag these sparse regions as the countryside (Noise/Outliers).
        """)
        
        st.write(
            "Below is a comparison demonstrating why K-Means fails on non-linear geometries (like interlocking Moons) "
            "due to its spherical centroid assumption, while DBSCAN clusters them perfectly by tracing density pathways."
        )
        
        # Walkthrough Steps
        st.markdown("##### 🤝 Side-by-Side Comparison: K-Means vs. DBSCAN")
        
        # Generate and Scale Moons Dataset
        from sklearn.datasets import make_moons
        from sklearn.preprocessing import StandardScaler
        
        X_moons_raw, y_moons_true = make_moons(n_samples=300, noise=0.08, random_state=42)
        scaler_moons = StandardScaler()
        X_moons_scaled = scaler_moons.fit_transform(X_moons_raw)
        
        # Fit K-Means
        km_moons = KMeans(n_clusters=2, random_state=42)
        labels_km = km_moons.fit_predict(X_moons_scaled)
        
        # Fit DBSCAN
        db_moons = DBSCAN(eps=0.3, min_samples=5)
        labels_db = db_moons.fit_predict(X_moons_scaled)
        
        # Setup DataFrames
        df_moons_km = pd.DataFrame(X_moons_scaled, columns=["x", "y"])
        df_moons_km["Cluster"] = [f"Cluster {l+1}" for l in labels_km]
        
        df_moons_db = pd.DataFrame(X_moons_scaled, columns=["x", "y"])
        df_moons_db["Cluster"] = [f"Cluster {l+1}" if l != -1 else "Noise" for l in labels_db]
        
        col_km_plot, col_db_plot = st.columns(2)
        
        with col_km_plot:
            st.markdown("###### **❌ K-Means Result (k=2)**")
            st.write("Struggles with non-linear borders and cuts the interlocking shapes in half trying to form spherical shapes:")
            
            fig_km_moons = px.scatter(
                df_moons_km, x="x", y="y", color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Set1,
                title="K-Means: Geometric Fail"
            )
            fig_km_moons.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)'
            )
            st.plotly_chart(fig_km_moons, use_container_width=True)
            
        with col_db_plot:
            st.markdown("###### **✔️ DBSCAN Result (ε=0.3, minPts=5)**")
            st.write("Traces continuous density pathways and naturally identifies two crescent clusters alongside outliers:")
            
            fig_db_moons = go.Figure()
            unique_db_labels = np.unique(labels_db)
            db_palette = px.colors.qualitative.Dark2
            
            for l in unique_db_labels:
                if l == -1:
                    df_n = df_moons_db[df_moons_db["Cluster"] == "Noise"]
                    fig_db_moons.add_trace(go.Scatter(
                        x=df_n["x"], y=df_n["y"], mode='markers',
                        marker=dict(symbol='x', size=8, color='#9CA3AF'),
                        name="Noise / Outliers"
                    ))
                else:
                    df_c = df_moons_db[df_moons_db["Cluster"] == f"Cluster {l+1}"]
                    color_h = db_palette[l % len(db_palette)]
                    fig_db_moons.add_trace(go.Scatter(
                        x=df_c["x"], y=df_c["y"], mode='markers',
                        marker=dict(symbol='circle', size=8, color=color_h),
                        name=f"Cluster {l+1}"
                    ))
            
            fig_db_moons.update_layout(
                title="DBSCAN: Density Success",
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)'
            )
            st.plotly_chart(fig_db_moons, use_container_width=True)
            
        st.markdown("##### 💻 Minimal Python Code Walkthrough:")
        st.code("""
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# 1. Generate interlocking moons dataset
X, _ = make_moons(n_samples=300, noise=0.08, random_state=42)

# 2. Standarize features (vital for DBSCAN epsilon radius consistency)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Instantiate and fit DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# Noise points are natively designated as -1
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Number of discovered density clusters: {n_clusters}")
        """, language="python")



# --- D. HIERARCHICAL PAGE ---
elif selected_algo == "🌲 Hierarchical (Agglomerative)":
    st.subheader("🌲 Hierarchical Clustering (Agglomerative)")
    tab_theory, tab_formula, tab_demo, tab_example = st.tabs(["📖 Theory", "🔢 Formula", "⚡ Live Demo", "💡 Example"])
    
    with tab_theory:
        st.markdown("""
        <div class="theory-box">
            <h4>Connectivity-Based Bottom-Up Agglomerative Hierarchy</h4>
            <p><strong>Agglomerative Hierarchical Clustering</strong> is a 'bottom-up' approach where each data point starts 
            as an independent cluster. Pairs of clusters are merged successively as you move up the hierarchy based on distance 
            until all belong to a single root cluster, generating a tree structure called a Dendrogram.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_th_l, col_th_r = st.columns(2)
        with col_th_l:
            st.markdown("##### 🌲 When to Choose Hierarchical Clustering:")
            st.markdown("""
            - You require a structured **taxonomy / tree hierarchy** (e.g., genetic groupings, file system indexing).
            - The **number of clusters is unknown** and you want to choose it visually after analyzing the tree.
            - You need reproducible clustering results (it is fully deterministic, unlike K-Means).
            """)
            
            st.success("""
            ✔️ **Hierarchical Strengths:**
            - Does not require a pre-determined cluster count.
            - Highly visual and interpretable thanks to the dendrogram structure.
            - Multiple distance metrics and merge definitions are supported.
            """)
            
            st.warning("""
            ❌ **Hierarchical Limitations:**
            - High time complexity $O(n^3)$ and space complexity $O(n^2)$ makes it struggle on large datasets.
            - Merges cannot be undone or adjusted retroactively.
            - Highly sensitive to outliers and noisy bridging nodes.
            """)
            
        with col_th_r:
            st.markdown("##### 🔗 The Four Linkage Criteria:")
            st.write("Linkage criteria define how the distance between two groups of points is computed:")
            
            with st.expander("💼 Ward Linkage (Minimize Variance)"):
                st.markdown("""
                - **Definition:** Merges the two clusters that result in the smallest increase in total within-cluster variance.
                - **Pros:** Highly effective at finding compact, clean, spherical clusters of similar sizes.
                - **Cons:** Strongly biased towards spherical shapes; only supports Euclidean distance.
                """)
                
            with st.expander("📏 Complete Linkage (Maximum Distance)"):
                st.markdown("""
                - **Definition:** Distance is defined as the maximum distance between a point in the first cluster and a point in the second.
                - **Pros:** Prevents overlapping boundaries; forms compact clusters with equal diameters.
                - **Cons:** Prone to the 'crowding' effect (forcing clusters apart due to single distant outliers).
                """)
                
            with st.expander("⚖️ Average Linkage (Mean Distance)"):
                st.markdown("""
                - **Definition:** Distance is the average of all pairwise distances between points in the two clusters.
                - **Pros:** Highly robust, stable, and less sensitive to extreme outliers than single or complete linkages.
                - **Cons:** Biased toward clusters of similar densities and sizes.
                """)
                
            with st.expander("⛓️ Single Linkage (Minimum Distance)"):
                st.markdown("""
                - **Definition:** Distance is the minimum distance between any point in the first cluster and any point in the second.
                - **Pros:** Natively captures highly curved, non-spherical, or interlocking cluster topologies.
                - **Cons:** Susceptible to the **chaining effect** (sparse bridging points cause massive distinct clusters to merge).
                """)
                
    with tab_formula:
        st.markdown("#### 🔢 Mathematical Formulations of Linkages")
        st.write(
            "Different linkages define different distance functions $d(u,v)$ between cluster $u$ and cluster $v$:"
        )
        
        st.markdown("##### 1. Ward Linkage Formula")
        st.latex(r"d(u,v) = \sqrt{\frac{2 n_u n_v}{n_u + n_v}} \| \bar{x}_u - \bar{x}_v \|")
        st.caption("📖 **Ward:** Computes the variance delta generated by merging cluster $u$ (size $n_u$, centroid $\bar{x}_u$) and cluster $v$ (size $n_v$, centroid $\bar{x}_v$).")
        
        st.markdown("##### 2. Complete Linkage Formula")
        st.latex(r"d(u,v) = \max_{x \in u, y \in v} d(x,y)")
        st.caption("📖 **Complete (Max):** Looks at the absolute maximum pairwise distance between the edges of clusters $u$ and $v$.")
        
        st.markdown("##### 3. Average Linkage Formula")
        st.latex(r"d(u,v) = \frac{1}{|u||v|} \sum_{x \in u} \sum_{y \in v} d(x,y)")
        st.caption("📖 **Average (Mean):** Averages all straight-line distances between every node of cluster $u$ and every node of cluster $v$.")
        
        st.markdown("##### 4. Single Linkage Formula")
        st.latex(r"d(u,v) = \min_{x \in u, y \in v} d(x,y)")
        st.caption("📖 **Single (Min):** Evaluates the absolute minimum distance separating the closest boundaries of clusters $u$ and $v$.")
        
    with tab_demo:
        st.markdown("#### ⚡ Live Hierarchical Sandbox")
        
        if st.session_state["X"] is not None:
            X = st.session_state["X"]
            df_raw = st.session_state["df_raw"]
            
            # Edge Case Warning for Single Linkage Chaining
            if hr_linkage == "single":
                st.warning("⚠️ **Single Linkage Chaining Alert:** Single linkage is highly susceptible to the 'chaining effect' where sparse trails of intermediate points merge major distinct clusters. Keep an eye on your cluster shapes!")
                
            # 1. Fit Agglomerative Model
            actual_metric = "euclidean" if hr_linkage == "ward" else hr_metric
            labels = fit_hierarchical(X, hr_n_clusters, hr_linkage, actual_metric)
            
            # 2. Compute Linkage Matrix
            Z = compute_linkage_matrix(X, hr_linkage, actual_metric)
            
            # 3. Main Sandbox Layout - 3 Columns
            col_scatter, col_dendro, col_metrics = st.columns([1.5, 1.5, 1])
            
            with col_scatter:
                st.markdown("##### 📍 Cluster Scatter Plot")
                df_clustered = pd.DataFrame(X, columns=["Feature 1 (Scaled)", "Feature 2 (Scaled)"])
                df_clustered["Cluster"] = [f"Cluster {l+1}" for l in labels]
                
                fig = px.scatter(
                    df_clustered, 
                    x="Feature 1 (Scaled)", 
                    y="Feature 2 (Scaled)", 
                    color="Cluster",
                    color_discrete_sequence=px.colors.qualitative.Dark2,
                    title=f"Hierarchical Results (k={hr_n_clusters})"
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.05)'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_dendro:
                st.markdown("##### 🌳 scipy Dendrogram & Cut-Line")
                
                # Calculate cut height dynamically
                if len(Z) >= hr_n_clusters:
                    merge_height = Z[len(Z) - hr_n_clusters, 2]
                    if len(Z) - hr_n_clusters + 1 < len(Z):
                        next_height = Z[len(Z) - hr_n_clusters + 1, 2]
                        cut_height = (merge_height + next_height) / 2.0
                    else:
                        cut_height = merge_height * 1.05
                else:
                    cut_height = 1.0
                
                # Plot Dendrogram in Matplotlib
                fig_dendro, ax = plt.subplots(figsize=(5, 4))
                sch.dendrogram(
                    Z, 
                    truncate_mode="lastp", 
                    p=30, 
                    show_leaf_counts=True, 
                    ax=ax,
                    leaf_rotation=90
                )
                ax.axhline(y=cut_height, color='#E02424', linestyle='--', linewidth=2, label=f"Cut at k={hr_n_clusters}")
                ax.set_title("Agglomerative Hierarchy Dendrogram")
                ax.set_xlabel("Data Points / Leaf Counts")
                ax.set_ylabel("Distance Height Threshold")
                ax.legend(loc="upper right")
                fig_dendro.patch.set_facecolor('none')
                ax.set_facecolor((0, 0, 0, 0.02))
                plt.tight_layout()
                st.pyplot(fig_dendro)
                plt.close(fig_dendro)
                
            with col_metrics:
                st.markdown("##### 📈 Model Metrics")
                
                # Calculate metrics
                sil = silhouette_score(X, labels)
                ch = calinski_harabasz_score(X, labels)
                db = davies_bouldin_score(X, labels)
                
                # Dynamic Session State Tracking for Delta Indicators
                prev_sil = st.session_state.get("hr_prev_sil", sil)
                prev_ch = st.session_state.get("hr_prev_ch", ch)
                prev_db = st.session_state.get("hr_prev_db", db)
                
                st.session_state["hr_prev_sil"] = sil
                st.session_state["hr_prev_ch"] = ch
                st.session_state["hr_prev_db"] = db
                
                delta_sil = sil - prev_sil
                delta_ch = ch - prev_ch
                delta_db = db - prev_db # lower is better, so a negative delta is good
                
                st.metric(
                    label="Silhouette Score", 
                    value=f"{sil:.4f}",
                    delta=f"{delta_sil:+.4f}" if abs(delta_sil) > 1e-6 else None,
                    help="Ranges from -1 to 1; higher is better."
                )
                st.metric(
                    label="Calinski-Harabasz Index", 
                    value=f"{ch:.2f}",
                    delta=f"{delta_ch:+.2f}" if abs(delta_ch) > 1e-6 else None,
                    help="Ratio of between-clusters variance to within-cluster variance. Higher is better."
                )
                st.metric(
                    label="Davies-Bouldin Score", 
                    value=f"{db:.4f}",
                    delta=f"{delta_db:+.4f}" if abs(delta_db) > 1e-6 else None,
                    delta_color="inverse", # Lower is better! A negative delta is colored green (good)
                    help="Average similarity measure of each cluster with its most similar cluster. Lower is better."
                )
                
                st.info("💡 **Dendrogram Tip:** The horizontal cut line represents where the tree is pruned to yield exactly your selected number of clusters $k$.")
        else:
            if selected_dataset == "Upload CSV":
                st.warning("⚠️ Please upload a valid CSV file first to activate the live clustering demo.")
            elif selected_dataset == "Enter Text":
                st.warning("⚠️ Please provide valid numeric coordinates first to activate the live clustering demo.")
                
    with tab_example:
        st.markdown("#### 💡 Worked Example: Iris Specimen Hierarchy")
        st.success("""
        🧬 **Visualizing the natural grouping:**
        The classic Iris dataset has 3 natural flower species. By passing the scaled Sepal Length and Sepal Width to 
        Hierarchical clustering, we can build a linkage tree and observe how biological sub-groups naturally split apart.
        """)
        
        # Load Iris features
        from sklearn.datasets import load_iris
        from sklearn.preprocessing import StandardScaler
        
        iris = load_iris()
        X_iris = iris.data[:, :2] # Sepal Length, Sepal Width
        scaler = StandardScaler()
        X_iris_scaled = scaler.fit_transform(X_iris)
        
        col_ex_plot, col_ex_dendro = st.columns(2)
        
        with col_ex_plot:
            st.markdown("###### **1. Agglomerative Cluster Output (k=3)**")
            model_ex = AgglomerativeClustering(n_clusters=3, linkage='ward')
            labels_ex = model_ex.fit_predict(X_iris_scaled)
            
            df_ex = pd.DataFrame(X_iris_scaled, columns=["Sepal Length (Scaled)", "Sepal Width (Scaled)"])
            df_ex["Species Cluster"] = [f"Flower Cluster {l+1}" for l in labels_ex]
            
            fig_ex = px.scatter(
                df_ex, x="Sepal Length (Scaled)", y="Sepal Width (Scaled)", color="Species Cluster",
                color_discrete_sequence=px.colors.qualitative.Set2,
                title="Iris Sepal Structure (Ward Linkage)"
            )
            fig_ex.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)'
            )
            st.plotly_chart(fig_ex, use_container_width=True)
            
        with col_ex_dendro:
            st.markdown("###### **2. Complete Linkage Dendrogram**")
            Z_iris = sch.linkage(X_iris_scaled, method='ward')
            
            fig_ex_d, ax_ex_d = plt.subplots(figsize=(6, 4))
            sch.dendrogram(Z_iris, truncate_mode="lastp", p=20, ax=ax_ex_d)
            # Find cut height for k=3
            merge_height = Z_iris[len(Z_iris) - 3, 2]
            next_height = Z_iris[len(Z_iris) - 2, 2]
            cut_height = (merge_height + next_height) / 2.0
            
            ax_ex_d.axhline(y=cut_height, color='#E02424', linestyle='--', linewidth=2, label="Split at k=3")
            ax_ex_d.set_title("Iris Hierarchical Taxonomy Tree")
            ax_ex_d.legend(loc="upper right")
            fig_ex_d.patch.set_facecolor('none')
            ax_ex_d.set_facecolor((0, 0, 0, 0.02))
            plt.tight_layout()
            st.pyplot(fig_ex_d)
            plt.close(fig_ex_d)
            
        st.markdown("##### 💻 Minimal Python Implementation Code:")
        st.code("""
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 1. Ingest Iris features
iris = load_iris()
X = iris.data[:, :2]

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Precompute Linkage Matrix for tree plotting
Z = sch.linkage(X_scaled, method='ward')

# 4. Instantiate and fit bottom-up Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X_scaled)
        """, language="python")



# --- E. GAUSSIAN MIXTURE MODEL PAGE ---
elif selected_algo == "🎲 Gaussian Mixture Model":
    st.subheader("🎲 Gaussian Mixture Model (GMM)")
    tab_theory, tab_formula, tab_demo, tab_example = st.tabs(["📖 Theory", "🔢 Formula", "⚡ Live Demo", "💡 Example"])
    
    with tab_theory:
        st.markdown("""
        <div class="theory-box">
            <h4>Probabilistic Density-Based Soft Clustering</h4>
            <p><strong>Gaussian Mixture Model (GMM)</strong> is a probabilistic model that assumes all data points are generated 
            from a mixture of a finite number of Gaussian (normal) distributions with unknown parameters. 
            Rather than drawing rigid borders, GMM outputs the probability that each point belongs to each Gaussian component.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_th_l, col_th_r = st.columns(2)
        with col_th_l:
            st.markdown("##### 🤝 Hard vs. Soft Clustering:")
            st.write(
                "Traditional models like K-Means perform **hard assignment**, forcing a data point to belong to exactly 1 cluster "
                "with 100% confidence. GMM introduces **soft assignment**, representing membership as a fractional probability."
            )
            
            st.info("""
            🎯 **Soft Assignment Concept:**
            Imagine segmenting e-commerce customers. In a hard clustering system, a customer is either classified as a *Premium Buyer* "
            "or a *Discount Hunter*. GMM allows a customer to be modeled as **70% Premium Buyer and 30% Discount Hunter**, "
            "reflecting real-world overlap!
            """)
            
            st.success("""
            ✔️ **GMM Strengths:**
            - Natively accommodates highly elliptical, elongated, and varying sized clusters.
            - Yields precise probabilistic cluster membership percentages.
            - Offers robust mathematical information criteria (BIC/AIC) to find the optimal component count.
            """)
            
        with col_th_r:
            st.markdown("##### 🔄 Expectation-Maximization (EM) Algorithm:")
            st.write(
                "GMM cannot be solved analytically. Instead, it is trained iteratively using the **Expectation-Maximization (EM)** algorithm:"
            )
            
            st.markdown("""
            1. **Initialization:** Component means ($\mu_k$), covariances ($\Sigma_k$), and mixing weights ($\pi_k$) are set (often using K-Means centroids).
            2. **E-Step (Expectation):** Calculates the **responsibilities** ($r_{ik}$) — the posterior probability that point $x_i$ belongs to Gaussian component $k$ under current parameters.
            3. **M-Step (Maximization):** Updates the GMM parameters ($\mu_k, \Sigma_k, \pi_k$) to maximize the expected log-likelihood of the data calculated in the E-step.
            4. **Convergence:** The steps repeat until the log-likelihood stabilizes or hits the max iteration limit.
            """)
            
            st.warning("""
            ❌ **GMM Limitations:**
            - Sensitive to initializations; can get trapped in sub-optimal local minima.
            - Susceptible to **degenerate covariance** anomalies where a component collapses into a zero-width spike.
            - Demands more computational memory and features scaling than K-Means.
            """)
            
    with tab_formula:
        st.markdown("#### 🔢 Mathematical Equations of GMM & EM")
        st.write(
            "The mathematical foundations under the GMM probability engine:"
        )
        
        st.markdown("##### 1. Mixture Density Function")
        st.latex(r"p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)")
        st.caption("📖 **Density Function:** Represents the overall probability of observing point $x$ as the sum of $K$ separate Gaussian densities $\mathcal{N}$ weighted by mixing coefficients $\pi_k$ (where $\sum \pi_k = 1$).")
        
        st.markdown("##### 2. Posterior Responsibility (E-Step)")
        st.latex(r"r_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}")
        st.caption("📖 **Responsibility $r_{ik}$:** Evaluates the relative probability that component $k$ was responsible for generating data point $x_i$.")
        
        st.markdown("##### 3. Objective Log-Likelihood Function")
        st.latex(r"\log L = \sum_{i=1}^{n} \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k) \right)")
        st.caption("📖 **Log-Likelihood:** The global objective function GMM seeks to maximize iteratively during M-step updates.")
        
    with tab_demo:
        st.markdown("#### ⚡ Live GMM Sandbox")
        
        if st.session_state["X"] is not None:
            X = st.session_state["X"]
            df_raw = st.session_state["df_raw"]
            
            # 1. Fit GMM Model
            labels, means, covariances, bic, aic, log_lik, converged = fit_gmm(
                X, gmm_n_components, gmm_covariance_type, gmm_max_iter
            )
            
            # Edge Case regularizer check for Degenerate Covariance
            is_degenerate = False
            if gmm_covariance_type == "spherical":
                if np.any(covariances < 1e-4):
                    is_degenerate = True
            elif gmm_covariance_type == "diag":
                if np.any(covariances < 1e-4):
                    is_degenerate = True
            elif gmm_covariance_type == "tied":
                if np.any(np.diag(covariances) < 1e-4):
                    is_degenerate = True
            else: # full
                for cov in covariances:
                    if np.any(np.diag(cov) < 1e-4):
                        is_degenerate = True
                        
            if is_degenerate:
                st.warning(
                    "⚠️ **Degenerate Covariance Alert:** One of the Gaussian components has collapsed into an extremely narrow line "
                    "or point (variance near zero). This often occurs when fitting too many components on small or highly sparse datasets. "
                    "Try reducing the number of components or selecting a regularized covariance structure like 'spherical' or 'diag'."
                )
                
            # 2. Main Sandbox Layout - 2 Columns
            col_scatter, col_metrics = st.columns([2, 1])
            
            with col_scatter:
                st.markdown("##### 📍 GMM Clustered Scatter with 1-σ Confidence Ellipses")
                df_clustered = pd.DataFrame(X, columns=["Feature 1 (Scaled)", "Feature 2 (Scaled)"])
                df_clustered["Cluster"] = [f"Component {l+1}" for l in labels]
                
                fig = go.Figure()
                colors_palette = px.colors.qualitative.Dark2
                
                # Plot components
                for comp in range(gmm_n_components):
                    df_c = df_clustered[df_clustered["Cluster"] == f"Component {comp+1}"]
                    color_hex = colors_palette[comp % len(colors_palette)]
                    
                    fig.add_trace(go.Scatter(
                        x=df_c["Feature 1 (Scaled)"],
                        y=df_c["Feature 2 (Scaled)"],
                        mode='markers',
                        marker=dict(symbol='circle', size=8, color=color_hex),
                        name=f"Component {comp+1}"
                    ))
                    
                    # Extract component covariance
                    if gmm_covariance_type == "tied":
                        comp_cov = covariances
                    else:
                        comp_cov = covariances[comp]
                        
                    comp_mean = means[comp]
                    
                    # Compute coordinates of 1-standard deviation ellipse
                    try:
                        ell_x, ell_y = get_ellipse_coords(comp_mean, comp_cov, n_std=1.0)
                        fig.add_trace(go.Scatter(
                            x=ell_x,
                            y=ell_y,
                            mode='lines',
                            line=dict(color=color_hex, width=2.5, dash='dash'),
                            name=f"1-σ Boundary {comp+1}",
                            showlegend=True
                        ))
                    except Exception:
                        pass
                        
                fig.update_layout(
                    title=f"GMM Fit Results ({gmm_covariance_type} covariance)",
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0.05)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
                )
                st.plotly_chart(fig, use_container_width=True)
                
            with col_metrics:
                st.markdown("##### 📈 Model Metrics")
                
                sil = silhouette_score(X, labels)
                
                # Dynamic Session State Tracking for Delta Indicators
                prev_bic = st.session_state.get("gmm_prev_bic", bic)
                prev_aic = st.session_state.get("gmm_prev_aic", aic)
                prev_log = st.session_state.get("gmm_prev_log", log_lik)
                prev_sil = st.session_state.get("gmm_prev_sil", sil)
                
                st.session_state["gmm_prev_bic"] = bic
                st.session_state["gmm_prev_aic"] = aic
                st.session_state["gmm_prev_log"] = log_lik
                st.session_state["gmm_prev_sil"] = sil
                
                delta_bic = bic - prev_bic
                delta_aic = aic - prev_aic
                delta_log = log_lik - prev_log
                delta_sil = sil - prev_sil
                
                st.metric(
                    label="BIC Score (Bayesian Info Criterion)", 
                    value=f"{bic:.2f}",
                    delta=f"{delta_bic:+.2f}" if abs(delta_bic) > 1e-2 else None,
                    delta_color="inverse", # Lower is better, so a negative delta is green (good)
                    help="Calculates model complexity vs log-likelihood fit. Lower is better."
                )
                st.metric(
                    label="AIC Score (Akaike Info Criterion)", 
                    value=f"{aic:.2f}",
                    delta=f"{delta_aic:+.2f}" if abs(delta_aic) > 1e-2 else None,
                    delta_color="inverse", # Lower is better, so a negative delta is green (good)
                    help="Information loss estimator. Lower is better."
                )
                st.metric(
                    label="Log-Likelihood Score", 
                    value=f"{log_lik:.2f}",
                    delta=f"{delta_log:+.2f}" if abs(delta_log) > 1e-2 else None,
                    help="Measures fit probability of overall dataset. Higher is better."
                )
                st.metric(
                    label="Silhouette Score (Hard Assign)", 
                    value=f"{sil:.4f}",
                    delta=f"{delta_sil:+.4f}" if abs(delta_sil) > 1e-4 else None,
                    help="Measures separation quality of hard labels. Higher is better."
                )
                
                if converged:
                    st.success("✔️ EM solver converged successfully.")
                else:
                    st.error("⚠️ EM solver failed to converge! Max iterations reached.")
                    
            # 3. BIC vs AIC Model Selection Plot
            st.markdown("---")
            st.markdown("### 🗺️ Model Selection Optimization: BIC vs. AIC Curves")
            st.write(
                "A standard method to find the optimal number of Gaussian components is to evaluate BIC and AIC across "
                "different component values. Look for the **'elbow'** or the absolute **minimum** of the curve. "
                "The red dashed line represents your active configuration."
            )
            
            with st.spinner("Calculating information criteria curve (k=1 to 10)..."):
                ks, bics, aics = compute_gmm_information_criteria(X, gmm_covariance_type, gmm_max_iter)
                
            df_ic = pd.DataFrame({
                "Components (k)": ks * 2,
                "Information Criteria": bics + aics,
                "Metric Type": ["BIC (Bayesian Information Criterion)"] * len(ks) + ["AIC (Akaike Information Criterion)"] * len(ks)
            })
            
            fig_ic = px.line(
                df_ic, 
                x="Components (k)", 
                y="Information Criteria", 
                color="Metric Type",
                markers=True,
                title="GMM Selection: BIC vs. AIC Curves (Lower is Better)"
            )
            fig_ic.add_vline(
                x=gmm_n_components, 
                line_dash="dash", 
                line_color="#E02424", 
                annotation_text=f"Active k={gmm_n_components}"
            )
            fig_ic.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.02)'
            )
            st.plotly_chart(fig_ic, use_container_width=True)
            
        else:
            if selected_dataset == "Upload CSV":
                st.warning("⚠️ Please upload a valid CSV file first to activate the live clustering demo.")
            elif selected_dataset == "Enter Text":
                st.warning("⚠️ Please provide valid numeric coordinates first to activate the live clustering demo.")
                
    with tab_example:
        st.markdown("#### 💡 Worked Example: Elongated Elliptical Blobs")
        
        st.success("""
        🗺️ **Real-World Analogy:**
        *GMM is like looking at weather patterns on a radar.* 
        If you look at storm clouds, they are rarely perfect circles. They stretch diagonally across regions 
        due to high-altitude winds (Anisotropic covariance). K-Means represents clouds as perfect circles, splitting 
        a single storm system down the middle because it cannot stretch. GMM models the clouds as elongated probability 
        ellipsoids, tracing their shapes perfectly.
        """)
        
        st.write(
            "Below is a comparison showing why K-Means fails on stretched diagonal distributions due to its isotropic "
            "spherical centroid boundary constraint, while GMM models them flawlessly."
        )
        
        st.markdown("##### 🤝 Side-by-Side Comparison: K-Means vs. GMM")
        
        # Generate stretched elongated synthetic blobs
        from sklearn.datasets import make_blobs
        from sklearn.preprocessing import StandardScaler
        
        X_raw, y_true = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=42)
        # Apply transformation matrix to stretch diagonal axis
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X_elongated = np.dot(X_raw, transformation)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_elongated)
        
        # Fit K-Means
        km = KMeans(n_clusters=2, random_state=42)
        labels_km = km.fit_predict(X_scaled)
        
        # Fit GMM
        gmm_ex = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
        labels_gmm = gmm_ex.fit_predict(X_scaled)
        means_gmm = gmm_ex.means_
        covs_gmm = gmm_ex.covariances_
        
        col_km, col_gmm = st.columns(2)
        
        with col_km:
            st.markdown("###### **❌ K-Means Result (k=2)**")
            st.write("Cuts the elongated diagonal shape perpendicularly because it assumes spherical shapes:")
            
            df_km = pd.DataFrame(X_scaled, columns=["x", "y"])
            df_km["Cluster"] = [f"Cluster {l+1}" for l in labels_km]
            
            fig_km = px.scatter(
                df_km, x="x", y="y", color="Cluster",
                color_discrete_sequence=px.colors.qualitative.Set1,
                title="K-Means: Spherical Constraint Failure"
            )
            fig_km.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)'
            )
            st.plotly_chart(fig_km, use_container_width=True)
            
        with col_gmm:
            st.markdown("###### **✔️ GMM Result (k=2, covariance='full')**")
            st.write("Stretches custom 1-σ ellipsoids along the diagonal axis, grouping points perfectly:")
            
            fig_gmm = go.Figure()
            df_gmm = pd.DataFrame(X_scaled, columns=["x", "y"])
            df_gmm["Cluster"] = [f"Component {l+1}" for l in labels_gmm]
            
            gmm_palette = px.colors.qualitative.Dark2
            for c in range(2):
                df_c = df_gmm[df_gmm["Cluster"] == f"Component {c+1}"]
                color_h = gmm_palette[c % len(gmm_palette)]
                
                fig_gmm.add_trace(go.Scatter(
                    x=df_c["x"], y=df_c["y"], mode='markers',
                    marker=dict(symbol='circle', size=8, color=color_h),
                    name=f"Component {c+1}"
                ))
                
                # Plot confidence ellipse
                ell_x, ell_y = get_ellipse_coords(means_gmm[c], covs_gmm[c], n_std=1.0)
                fig_gmm.add_trace(go.Scatter(
                    x=ell_x, y=ell_y, mode='lines',
                    line=dict(color=color_h, width=2.5, dash='dash'),
                    name=f"1-σ Ellipse {c+1}"
                ))
                
            fig_gmm.update_layout(
                title="GMM: Elliptical Topography Success",
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.05)'
            )
            st.plotly_chart(fig_gmm, use_container_width=True)
            
        st.markdown("##### 💻 Minimal Python Implementation Code:")
        st.code("""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# 1. Generate elongated elliptical blob dataset
X_raw, _ = make_blobs(n_samples=300, centers=2, cluster_std=1.0, random_state=42)
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
X_elongated = np.dot(X_raw, transformation)

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_elongated)

# 3. Instantiate and fit Gaussian Mixture Model
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
gmm.fit(X_scaled)

# 4. Extract soft probabilities and hard label assignments
hard_labels = gmm.predict(X_scaled)
soft_probabilities = gmm.predict_proba(X_scaled)

# Extract fitted parameters
means = gmm.means_
covariances = gmm.covariances_
        """, language="python")


# --- F. COMPARE ALL ALGORITHMS PAGE ---
elif selected_algo == "📊 Compare All Algorithms":
    st.subheader("📊 Compare All Algorithms Simultaneously")
    
    if st.session_state["X"] is not None:
        X = st.session_state["X"]
        df_raw = st.session_state["df_raw"]
        
        # Run all algorithms with a high-fidelity spinner
        with st.spinner("Running K-Means, DBSCAN, Hierarchical & Gaussian Mixture simultaneously..."):
            results = run_all_algorithms(X, comp_k, comp_eps, comp_min_samples)
            
        # Section 1 — Side-by-side Cluster Plots
        st.markdown(f"### 📍 Side-by-Side Cluster Plots: {selected_dataset} ({len(X)} samples)")
        st.write(
            "Compare the physical shapes and partitions generated by each model on the exact same scaled features in real-time."
        )
        
        col_grid1, col_grid2 = st.columns(2)
        algos_keys = list(results.keys())
        
        for idx, name in enumerate(algos_keys):
            target_col = col_grid1 if idx % 2 == 0 else col_grid2
            res = results[name]
            
            with target_col:
                sil_val = res["silhouette"]
                sil_str = f"{sil_val:.4f}" if isinstance(sil_val, float) else "N/A"
                st.markdown(f"##### **{name}** (Silhouette: `{sil_str}`)")
                
                if res["error"] or res["labels"] is None:
                    # Draw a neutral gray plot representing failure
                    fig = px.scatter(
                        df_raw, 
                        x=df_raw.columns[0], 
                        y=df_raw.columns[1],
                        color_discrete_sequence=["#888888"],
                        title=f"{name} Fit Failed"
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.05)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.warning(f"⚠️ **{name} Failure:** Could not cluster dataset with selected parameters.")
                else:
                    df_c = pd.DataFrame(X, columns=["Feature 1 (Scaled)", "Feature 2 (Scaled)"])
                    labels = res["labels"]
                    
                    # Convert labels to human-readable strings (especially DBSCAN noise)
                    cluster_labels = []
                    for l in labels:
                        if l == -1:
                            cluster_labels.append("Noise / Outliers")
                        else:
                            cluster_labels.append(f"Cluster {l+1}")
                    df_c["Cluster"] = cluster_labels
                    
                    fig = px.scatter(
                        df_c, 
                        x="Feature 1 (Scaled)", 
                        y="Feature 2 (Scaled)", 
                        color="Cluster",
                        color_discrete_sequence=px.colors.qualitative.Dark2,
                        title=f"{name} Clustered Output"
                    )
                    fig.update_layout(
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0.03)',
                        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
        # Section 2 — Metrics Comparison Table
        st.markdown("---")
        st.markdown("### 📊 Consolidated Metrics Comparison Table")
        st.write(
            "The table compiles fundamental statistical attributes across all fitted models. "
            "The **best value** in each metric column is highlighted in **green**."
        )
        
        comp_rows = []
        for name, res in results.items():
            sil = res["silhouette"]
            ch = res["ch"]
            db = res["db"]
            t_ms = res["time"]
            
            comp_rows.append({
                "Algorithm": name,
                "Clusters": int(res["n_clusters"]),
                "Silhouette ↑": f"{sil:.4f}" if isinstance(sil, float) else "N/A",
                "Calinski-Harabasz ↑": f"{ch:.2f}" if isinstance(ch, float) else "N/A",
                "Davies-Bouldin ↓": f"{db:.4f}" if isinstance(db, float) else "N/A",
                "Fit Time (ms)": f"{t_ms:.2f}" if isinstance(t_ms, float) else "N/A"
            })
            
        df_metrics = pd.DataFrame(comp_rows)
        
        # Style best values in green
        def highlight_best_metrics(df):
            style_df = pd.DataFrame('', index=df.index, columns=df.columns)
            green_style = 'background-color: rgba(34, 197, 94, 0.25); color: #15803d; font-weight: bold; border: 1px solid rgba(34, 197, 94, 0.4);'
            
            # Silhouette: Higher is better
            sil_vals = pd.to_numeric(df["Silhouette ↑"], errors='coerce')
            if not sil_vals.dropna().empty:
                style_df.loc[sil_vals.idxmax(), "Silhouette ↑"] = green_style
                
            # Calinski-Harabasz: Higher is better
            ch_vals = pd.to_numeric(df["Calinski-Harabasz ↑"], errors='coerce')
            if not ch_vals.dropna().empty:
                style_df.loc[ch_vals.idxmax(), "Calinski-Harabasz ↑"] = green_style
                
            # Davies-Bouldin: Lower is better
            db_vals = pd.to_numeric(df["Davies-Bouldin ↓"], errors='coerce')
            if not db_vals.dropna().empty:
                style_df.loc[db_vals.idxmin(), "Davies-Bouldin ↓"] = green_style
                
            # Fit Time: Lower is better
            time_vals = pd.to_numeric(df["Fit Time (ms)"], errors='coerce')
            if not time_vals.dropna().empty:
                style_df.loc[time_vals.idxmin(), "Fit Time (ms)"] = green_style
                
            return style_df
            
        styled_df = df_metrics.style.apply(highlight_best_metrics, axis=None)
        st.dataframe(styled_df, use_container_width=True)
        
        # Section 3 — Radar / Spider Chart
        st.markdown("---")
        st.markdown("### 🕸️ Joint Model Performance: Radar / Spider Comparison")
        st.write(
            "The radar chart visualizes the relative strengths of each algorithm across four fundamental dimensions. "
            "A wider, filled area represents a more versatile or optimal model for the current dataset."
        )
        
        categories = ['Silhouette Score', 'Fitting Speed', 'Handles Outliers / Noise', 'Shape Flexibility']
        fig_radar = go.Figure()
        
        radar_capacities = {
            "K-Means": {"noise": 0.1, "shape": 0.1},
            "DBSCAN": {"noise": 1.0, "shape": 1.0},
            "Hierarchical (Agglomerative)": {"noise": 0.3, "shape": 0.5},
            "Gaussian Mixture Model": {"noise": 0.7, "shape": 0.9}
        }
        
        for name, res in results.items():
            # Silhouette norm: maps [-1, 1] to [0, 1]
            sil = res["silhouette"]
            if isinstance(sil, float):
                sil_norm = max(0.0, (sil + 1.0) / 2.0)
            else:
                sil_norm = 0.0
                
            # Fit time norm: maps 0ms -> 1.0, and decays smoothly
            t = res["time"]
            if isinstance(t, float):
                speed_norm = 1.0 / (1.0 + t / 5.0)
            else:
                speed_norm = 0.0
                
            noise_val = radar_capacities[name]["noise"]
            shape_val = radar_capacities[name]["shape"]
            
            # Close the loop mathematically
            r_values = [sil_norm, speed_norm, noise_val, shape_val, sil_norm]
            theta_categories = categories + [categories[0]]
            
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values,
                theta=theta_categories,
                fill='toself',
                name=name
            ))
            
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor="rgba(0,0,0,0.1)",
                    angle=45
                ),
                angularaxis=dict(
                    gridcolor="rgba(0,0,0,0.1)"
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Section 4 — "Which Algorithm Should I Use?" Decision Guide
        st.markdown("---")
        st.markdown("### 🧭 Interactive Clustering Flowchart & Decision Support")
        
        with st.expander("🤔 Not sure which to pick? Open the Decision Guide"):
            st.markdown("#### 🚀 Algorithm Selection Flowchart")
            st.markdown("""
            Use this interactive matrix and decision flowchart to select the ideal clustering model based on your dataset properties:
            """)
            
            st.markdown("""
            | Primary Question | Action / Recommendation | Analytical Rationale |
            | :--- | :--- | :--- |
            | **1. Do you know the target cluster count ($k$)?** | **YES** $\to$ Proceed to next query. <br> **NO** $\to$ Select **DBSCAN** or **Hierarchical**. | DBSCAN detects density peaks automatically. Hierarchical builds a tree structure showing splits. |
            | **2. Does your data have heavy noise/outliers?** | **YES** $\to$ Choose **DBSCAN**. <br> **NO** $\to$ Proceed to next query. | DBSCAN filters outliers seamlessly. K-Means shifts centroids and fails. |
            | **3. Are clusters elliptical or overlapping?** | **YES** $\to$ Use **Gaussian Mixture Model (GMM)**. <br> **NO** $\to$ Choose **K-Means** or **Hierarchical**. | GMM uses covariance matrices to model anisotropic ellipses and yields soft probabilities. |
            | **4. Do you need a taxonomic tree structure?** | **YES** $\to$ Select **Hierarchical (Agglomerative)**. <br> **NO** $\to$ Proceed with **K-Means**. | Hierarchical generates deterministic branching trees (Dendrograms). |
            """)
            
            st.markdown("##### 🗺️ Textual Decision Flow")
            st.code("""
[START: Clustering Task]
   |
   |---> Do you know k?
            |-- NO  --> [DBSCAN] (Density-based) or [Hierarchical] (Build dendrogram)
            |-- YES --> Do you have severe noise/outliers?
                          |-- YES --> [DBSCAN] (Filters noise natively)
                          |-- NO  --> Are clusters elliptical or overlapping?
                                        |-- YES --> [GMM] (Soft probability ellipsoids)
                                        |-- NO  --> [K-Means] (Fast, spherical centroids)
            """)
            
        # Section 5 — Evaluation Metrics Explained
        st.markdown("### 📚 Evaluation Metrics Explained")
        
        with st.expander("📖 1. Silhouette Score"):
            st.markdown("#### **Silhouette Score**")
            st.write(
                "The Silhouette Coefficient evaluates how well-defined and compact the clusters are. "
                "It scores each point based on its compaction within its own cluster compared to its separation from others."
            )
            st.latex(r"s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}")
            st.markdown("""
            * **$a(i)$**: The mean intra-cluster distance between point $i$ and all other points in the same cluster (compaction).
            * **$b(i)$**: The mean distance from point $i$ to the nearest cluster of which it is not a part (separation).
            * **Interpretation:**
              * **$+1.0$**: Ideal score; indicates clusters are dense, compact, and completely separated.
              * **$0.0$**: Indicates overlapping boundaries or points sitting directly on decision borders.
              * **$-1.0$**: Indicates highly misclustered structures where points are placed in the wrong groups.
            """)
            
        with st.expander("📖 2. Calinski-Harabasz Index (Variance Ratio Criterion)"):
            st.markdown("#### **Calinski-Harabasz Index**")
            st.write(
                "Evaluates the ratio of sum of between-cluster dispersion to within-cluster dispersion. "
                "It represents how dispersed clusters are from each other vs. how tight they are internally."
            )
            st.latex(r"s = \frac{\text{Tr}(B_k)}{\text{Tr}(W_k)} \times \frac{N - k}{k - 1}")
            st.markdown("""
            * **$B_k$**: Between-cluster dispersion matrix.
            * **$W_k$**: Within-cluster dispersion matrix.
            * **$N$**: Total sample count; **$k$**: Number of clusters.
            * **Interpretation:**
              * **Higher is Better:** A higher index indicates the between-cluster variance is large (clusters are far apart) "
              "while the within-cluster variance is small (clusters are dense and compact).
            """)
            
        with st.expander("📖 3. Davies-Bouldin Score"):
            st.markdown("#### **Davies-Bouldin Score**")
            st.write(
                "Calculates the average similarity between each cluster and its most similar counterpart. "
                "Similarity is defined as a ratio of cluster sizes (dispersions) to the distance separating their centroids."
            )
            st.latex(r"R_{ij} = \frac{s_i + s_j}{d(\mu_i, \mu_j)}")
            st.latex(r"DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} R_{ij}")
            st.markdown("""
            * **$s_i$**: The average distance of all points in cluster $i$ to their centroid (spread).
            * **$d(\mu_i, \mu_j)$**: The Euclidean distance separating cluster centroids $i$ and $j$.
            * **Interpretation:**
              * **Lower is Better:** The absolute minimum score is $0.0$. Lower scores represent clusters that are highly dense (low $s$) "
              "and widely separated from each other (high centroid distance $d$).
            """)
            
        with st.expander("📖 4. Inertia / Within-Cluster Sum of Squares (WCSS)"):
            st.markdown("#### **Inertia (Within-Cluster Sum of Squares)**")
            st.write(
                "Represents the absolute sum of squared Euclidean distances of all points in a cluster to their respective centroid. "
                "This metric is the objective function that **K-Means** seeks to minimize during training."
            )
            st.latex(r"WCSS = \sum_{j=1}^{k} \sum_{x_i \in C_j} \| x_i - \mu_j \|^2")
            st.markdown("""
            * **$\mu_j$**: Centroid coordinate of cluster $C_j$.
            * **$\| x_i - \mu_j \|^2$**: Squared Euclidean distance of sample $x_i$ to its centroid.
            * **Crucial Limitations:**
              * **Only applies to K-Means:** It is not valid or comparable for density-based DBSCAN, tree-like Hierarchical, or soft-distribution GMMs.
              * **Scale Sensitive:** It is highly sensitive to features scaling, outliers, and always decreases as you increase $k$ (forcing the use of the Elbow Method).
            """)
            
    else:
        st.warning("⚠️ Please load a valid dataset first to compare all clustering algorithms simultaneously.")


