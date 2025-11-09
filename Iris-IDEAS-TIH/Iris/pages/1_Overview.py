import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
import pdf_export
st.title("📘 Iris Dataset Overview")

@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    return df, iris.target_names

df, target_names = load_iris_data()

st.markdown("""
### 🌼 About the Dataset
The **Iris dataset** is a classic dataset in machine learning, introduced by *Ronald A. Fisher* (1936).
It contains **150 samples** from three species:
- Setosa (0)
- Versicolor (1)
- Virginica (2)
""")

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.subheader("📏 Basic Statistics")
st.write(df.describe())


# --- PDF Export Section ---
st.markdown("---")
st.subheader("📄 Export This Page")
st.markdown("You can export **all content** of this page (description + preview + summary) as a single PDF file.")

if st.button("📤 Export Full Page to PDF"):
    try:
        # Create a simple "figures" list for pdf_export
        # We'll generate text-based figures using matplotlib to hold content
        import matplotlib.pyplot as plt

        figs_to_export = []

        # 1️⃣ Dataset Description as figure
        fig_desc, ax_desc = plt.subplots(figsize=(8, 6))
        ax_desc.axis("off")
        description_text = (
            "🌼 About the Dataset\n\n"
            "The Iris dataset is a classic dataset in machine learning, introduced by Ronald A. Fisher (1936).\n"
            "It contains 150 samples from three species:\n"
            "- Setosa (0)\n"
            "- Versicolor (1)\n"
            "- Virginica (2)\n"
        )
        ax_desc.text(0, 0.5, description_text, fontsize=12, va='top')
        figs_to_export.append(fig_desc)

        # 2️⃣ Dataset Preview as figure
        fig_preview, ax_preview = plt.subplots(figsize=(10, 6))
        ax_preview.axis("off")
        ax_preview.table(cellText=df.head(15).values,
                         colLabels=df.head(0).columns,
                         loc='center',
                         cellLoc='center')
        ax_preview.set_title("📄 Dataset Preview (first 15 rows)")
        figs_to_export.append(fig_preview)

        # 3️⃣ Summary Statistics as figure
        fig_stats, ax_stats = plt.subplots(figsize=(10, 6))
        ax_stats.axis("off")
        ax_stats.table(cellText=df.describe().round(2).values,
                       colLabels=df.describe().columns,
                       rowLabels=df.describe().index,
                       loc='center',
                       cellLoc='center')
        ax_stats.set_title("📏 Basic Statistics")
        figs_to_export.append(fig_stats)

        # Export all figures as one PDF
        pdf_path = pdf_export.create_multi_fig_pdf(figs_to_export, title="Iris Overview - Full Page Export")
        st.success(f"✅ PDF exported successfully! Saved to: {pdf_path}")
        st.info("You can find it inside the Iris/exports folder.")

    except Exception as e:
        st.error(f"❌ Export failed: {e}")