# -----------------------------------------------------------------------------
# 🔬 ClusterLab — data_loader.py
# -----------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris
from sklearn.preprocessing import StandardScaler

def load_builtin_dataset(name):
    """
    Loads one of the predefined scikit-learn datasets.
    Returns: (X_scaled, df_raw)
    """
    if name == "Blobs (Easy)":
        X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)
        df_raw = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        df_raw["Target"] = y
        
    elif name == "Moons (Non-linear)":
        X, y = make_moons(n_samples=300, noise=0.05, random_state=42)
        df_raw = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        df_raw["Target"] = y
        
    elif name == "Circles (Nested)":
        X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)
        df_raw = pd.DataFrame(X, columns=["Feature 1", "Feature 2"])
        df_raw["Target"] = y
        
    elif name == "Iris (Real-world)":
        iris = load_iris()
        # Use first 2 features for 2D plotting (sepal length, sepal width)
        X = iris.data[:, :2]
        feature_names = [c.replace(" (cm)", "") for c in iris.feature_names[:2]]
        df_raw = pd.DataFrame(X, columns=feature_names)
        df_raw["Target"] = iris.target
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        df_raw["Species"] = df_raw["Target"].map(species_map)
        
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_raw

def load_csv_dataset():
    """
    Renders st.file_uploader and handles CSV parsing, column selection,
    and automatic scaling.
    Returns: (X_scaled, df_raw_preview_or_none) or (None, None)
    """
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Show preview
            st.markdown("##### 📄 CSV File Preview (First 5 rows)")
            st.dataframe(df.head(5), use_container_width=True)
            
            # Column selection
            columns = df.columns.tolist()
            if len(columns) < 2:
                st.warning("⚠️ The uploaded CSV must have at least 2 columns to perform 2D clustering analysis.")
                return None, None
                
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X axis column", options=columns, index=0)
            with col2:
                remaining_cols = [c for c in columns if c != x_col]
                y_col = st.selectbox("Select Y axis column", options=remaining_cols, index=0)
            
            # Validate columns are numeric
            if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
                st.warning("⚠️ Both selected columns must contain numeric values only. Please re-select.")
                return None, None
                
            # Filter and drop NaNs
            df_subset = df[[x_col, y_col]].dropna()
            
            if len(df_subset) < 5:
                st.warning("⚠️ The dataset contains too few valid numeric rows after removing missing values.")
                return None, None
                
            X = df_subset.values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            df_raw = pd.DataFrame(X, columns=[x_col, y_col])
            
            return X_scaled, df_raw
            
        except Exception as e:
            st.warning(f"⚠️ Error reading CSV file: {str(e)}")
            return None, None
            
    return None, None

def load_text_dataset():
    """
    Renders st.text_area and parses text input as numeric pairs.
    Returns: (X_scaled, df_raw) or (None, None)
    """
    default_text = (
        "1.0, 2.0\n"
        "1.5, 1.8\n"
        "1.2, 2.2\n"
        "5.0, 8.0\n"
        "5.2, 7.8\n"
        "4.8, 8.2\n"
        "8.0, 2.0\n"
        "8.3, 2.3\n"
        "7.8, 1.8\n"
        "8.1, 2.1"
    )
    
    st.markdown("##### 📝 Paste Comma-Separated Numeric Coordinates")
    st.caption("Enter one coordinate pair per line, separated by a comma (e.g. `X, Y`):")
    
    text_input = st.text_area(
        "Data Points (2D)", 
        value=default_text, 
        height=180,
        placeholder="e.g.\n1.0, 2.0\n3.5, 4.2"
    )
    
    if not text_input.strip():
        st.warning("⚠️ The text area is empty. Please enter numeric coordinate pairs.")
        return None, None
        
    try:
        lines = text_input.strip().split("\n")
        data_list = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                st.warning(f"⚠️ Line {i+1} does not have exactly 2 columns: '{line}'")
                return None, None
            
            try:
                x_val = float(parts[0].strip())
                y_val = float(parts[1].strip())
                data_list.append([x_val, y_val])
            except ValueError:
                st.warning(f"⚠️ Line {i+1} has invalid non-numeric values: '{line}'")
                return None, None
                
        if len(data_list) < 3:
            st.warning("⚠️ Please provide at least 3 valid numeric coordinate rows for clustering.")
            return None, None
            
        X = np.array(data_list)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        df_raw = pd.DataFrame(X, columns=["Feature X", "Feature Y"])
        return X_scaled, df_raw
        
    except Exception as e:
        st.warning(f"⚠️ Parsing error: {str(e)}")
        return None, None
