
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from sklearn.decomposition import PCA

# Load models
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('xgb_model.pkl')

# Page config
st.set_page_config(page_title="DDoS Attack Detector", page_icon="🛡️", layout="wide")

# Header
st.title("🛡️ DDoS Attack Detection System")
st.markdown("### Machine Learning-Based Profiling of DDoS Attacks on IoT Network Traffic")
st.markdown("---")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("""
**Model:** XGBoost + PCA  
**Technique:** Dimensionality Reduction via PCA  
**Purpose:** DDoS Detection in IoT Traffic  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Nuruldelmia Binti Idris**")
st.sidebar.markdown("University of Malaya FYP 2026")
st.sidebar.markdown("Final Year Project")
st.sidebar.markdown("Machine Learning-Based Profiling of DDoS Attacks on Network Traffic in Internet of Things Environments")

# Tabs
tab1, tab2 = st.tabs(["🔍 Detection", "📊 PCA Explanation"])

# ─── TAB 1: DETECTION ────────────────────────────────────────────────────
with tab1:
    st.subheader("📂 Upload Network Traffic CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded! Shape: {df.shape}")

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        has_label = 'label' in df.columns

        if has_label:
            df_filtered = df[df['label'].str.contains('DDoS|BenignTraffic', case=False, na=False)].copy()
            y_true = (df_filtered['label'].str.contains('DDoS', case=False)).astype(int)
            X = df_filtered.drop(columns=['label'])
        else:
            X = df.copy()

        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)

        if st.button("🔍 Detect DDoS Attacks"):
            with st.spinner("Analyzing traffic..."):
                X_scaled = scaler.transform(X)
                X_pca = pca.transform(X_scaled)
                predictions = model.predict(X_pca)
                proba = model.predict_proba(X_pca)

            st.markdown("---")
            st.subheader("🎯 Detection Results")

            col1, col2, col3 = st.columns(3)
            total = len(predictions)
            ddos_count = predictions.sum()
            benign_count = total - ddos_count

            col1.metric("Total Samples", total)
            col2.metric("🚨 DDoS Detected", int(ddos_count))
            col3.metric("✅ Benign Traffic", int(benign_count))

            st.subheader("📈 Traffic Distribution")
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            ax1.pie([benign_count, ddos_count],
                    labels=['Benign', 'DDoS'],
                    colors=['seagreen', 'crimson'],
                    autopct='%1.1f%%', startangle=90)
            ax1.set_title('Traffic Classification')
            st.pyplot(fig1)

            if has_label:
                st.subheader("📉 Confusion Matrix")
                cm = confusion_matrix(y_true, predictions)
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Benign', 'DDoS'],
                            yticklabels=['Benign', 'DDoS'])
                ax2.set_ylabel('Actual')
                ax2.set_xlabel('Predicted')
                ax2.set_title('Confusion Matrix')
                st.pyplot(fig2)

                st.subheader("📋 Classification Report")
                report = classification_report(y_true, predictions,
                                               target_names=['Benign', 'DDoS'],
                                               output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().round(4))

                st.subheader("📊 Cohen's Kappa Score")
                kappa = cohen_kappa_score(y_true, predictions)
                col4, col5 = st.columns(2)
                col4.metric("Cohen's Kappa", round(kappa, 4))
                col5.metric("Interpretation",
                           "Almost Perfect" if kappa > 0.8 else "Substantial" if kappa > 0.6 else "Moderate")

            st.success("✅ Analysis Complete!")

    else:
        st.info("👆 Please upload a CSV file to begin detection.")
        st.markdown("---")
        st.subheader("ℹ️ How to use")
        st.markdown("""
        1. Upload a network traffic CSV file
        2. Click **Detect DDoS Attacks**
        3. View detection results and charts
        """)

# ─── TAB 2: PCA EXPLANATION ──────────────────────────────────────────────
with tab2:
    st.subheader("📊 How PCA Works — Step by Step")
    st.markdown("PCA reduced **46 original features → 19 principal components** explaining **95.2% variance**")
    st.markdown("---")

    # Step 1
    st.markdown("### Step 1 — Standardize the Data")
    st.info("Scale all features to mean=0 and std=1 so no single feature dominates.")
    st.code("Z = (X - mean) / std", language='python')

    # Step 2
    st.markdown("### Step 2 — Covariance Matrix")
    st.info("Measures how features vary together. Result: 46×46 matrix.")
    st.image('step2_covariance.png')

    # Step 3
    st.markdown("### Step 3 — Eigenvalues & Eigenvectors")
    st.info("Eigenvalues show importance of each direction. Sorted from largest to smallest.")
    st.image('step3_eigenvalues.png')

    # Step 4
    st.markdown("### Step 4 — Select Components at 95% Variance")
    st.info("19 components needed to explain 95.2% of total variance.")
    st.image('step4_variance.png')

    # Step 5 — Why 19 table
    st.markdown("### Step 5 — Why 19 Components?")
    data = {
        'Components': list(range(15, 25)),
        'Cumulative Variance (%)': [91.15, 92.28, 93.35, 94.31, 95.20,
                                     96.02, 96.75, 97.41, 97.99, 98.51],
        'Selected': ['', '', '', '', '✅ Selected (≥95%)',
                     '', '', '', '', '']
    }
    st.dataframe(pd.DataFrame(data))

    # Step 6
    st.markdown("### Step 6 — Final Transformation")
    col1, col2, col3 = st.columns(3)
    col1.metric("Original Features", "46")
    col2.metric("PCA Components", "19")
    col3.metric("Variance Retained", "95.2%")

    st.success("PCA successfully reduced dimensionality by 58.7% while retaining 95.2% of information! ✅")

print("App file created ✅")
