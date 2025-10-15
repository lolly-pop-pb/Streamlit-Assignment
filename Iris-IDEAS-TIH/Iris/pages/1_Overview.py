import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

st.title("📘 Iris Dataset Overview")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

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
