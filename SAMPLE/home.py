# Home.py
import streamlit as st

st.set_page_config(
    page_title="Iris Dataset App",
    layout="wide",
    page_icon="🌸"
)

st.title("🌸 Welcome to the Iris Dataset Explorer")

st.markdown("""
This application demonstrates data analysis and machine learning using the **Iris dataset**.  
Navigate using the sidebar to explore:
1. **Dataset Overview** — Learn about the data and its features  
2. **Exploratory Data Analysis (EDA)** — Visualize relationships and distributions  
3. **Machine Learning Models** — Train and test classifiers  
4. **DuckDB SQL Playground** — Query the dataset using SQL  
""")