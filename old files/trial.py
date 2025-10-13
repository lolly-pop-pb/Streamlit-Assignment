import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# -----------------------------------------------
# Load dataset
# -----------------------------------------------
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['target'] = iris_data.target
target_names = iris_data.target_names

st.set_page_config(page_title="Iris Dataset App")

# -----------------------------------------------
# Tabs
# -----------------------------------------------
tab1, tab2, tab3 = st.tabs(["📘 Dataset Overview", "📊 Exploratory Data Analysis", "🤖 ML Models"])

# -----------------------------------------------
# TAB 1: Overview
# -----------------------------------------------
with tab1:
    st.title("🌸 Iris Dataset Overview")

    st.markdown("""
    The **Iris dataset** is a classic dataset in machine learning and statistics.
    It contains 150 samples of iris flowers — each belonging to one of three species:
    **Setosa**, **Versicolor**, and **Virginica**.
    
    **Features:**
    - *Sepal length (cm)*  
    - *Sepal width (cm)*  
    - *Petal length (cm)*  
    - *Petal width (cm)*  

    **Target:**
    - 0 → Setosa  
    - 1 → Versicolor  
    - 2 → Virginica  

    **Source:**  
    - Collected by British biologist **Ronald A. Fisher** (1936).  
    - Available via `sklearn.datasets.load_iris()`.
    """)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📏 Basic Statistics")
    st.write(df.describe())

# -----------------------------------------------
# TAB 2: EDA
# -----------------------------------------------
with tab2:
    st.title("📊 Exploratory Data Analysis")
    st.markdown("Here we explore the distribution and relationships among the Iris dataset features.")

    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x='target', data=df, palette='viridis', ax=ax)
    ax.set_xticklabels(target_names)
    st.pyplot(fig)

    st.subheader("Pairplot of Features")
    st.markdown("Visualize relationships among features colored by species.")
    pairplot_fig = sns.pairplot(df, hue="target", palette="husl", diag_kind="hist")
    st.pyplot(pairplot_fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


    # NEW STUFF
    features = iris_data.feature_names
    st.subheader("Feature Histograms")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i, feature in enumerate(features):
        row, col = divmod(i, 2)
        sns.histplot(df[feature], kde=True, color='teal', ax=axes[row, col])
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)



    # --- 2. Line Chart (Mean of features per species) ---
    st.subheader("Mean of Each Feature per Species (Line Chart)")
    mean_df = df.groupby("target")[iris_data.feature_names].mean().T
    mean_df.columns = target_names
    fig, ax = plt.subplots(figsize=(6, 4))
    mean_df.plot(kind="line", ax=ax, marker='o')
    ax.set_title("Feature Means by Species")
    ax.set_ylabel("Mean Value (cm)")
    st.pyplot(fig)

    # --- 3. Boxplot ---
    st.subheader("Boxplot: Feature Distributions by Species")
    feature = st.selectbox("Select a feature for boxplot:", iris_data.feature_names)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="target", y=feature, data=df, palette="Set2", ax=ax)
    ax.set_xticklabels(target_names)
    ax.set_title(f"{feature} by Species")
    st.pyplot(fig)

    # --- 4. Violin Plot (optional) ---
    st.subheader("Violin Plot: Detailed Feature Distribution")
    feature_v = st.selectbox("Select a feature for violin plot:", iris_data.feature_names, key="violin")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(x="target", y=feature_v, data=df, palette="coolwarm", ax=ax)
    ax.set_xticklabels(target_names)
    ax.set_title(f"{feature_v} by Species")
    st.pyplot(fig)

    

# -----------------------------------------------
# TAB 3: Machine Learning Models
# -----------------------------------------------
with tab3:
    st.title("🤖 Machine Learning Models")

    st.write("Choose a regression model to predict one of the features (e.g., *petal width*) from the others.")

    # Select target feature
    target_feature = st.selectbox("Select the feature to predict:", iris_data.feature_names)
    X = df.drop(columns=['target', target_feature])
    y = df[target_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    col1, col2 = st.columns(2)

    # Linear Regression
    with col1:
        st.subheader("📈 Linear Regression")
        if st.button("Run Linear Regression"):
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred_lr = lr.predict(X_test)
            r2 = r2_score(y_test, y_pred_lr)
            mse = mean_squared_error(y_test, y_pred_lr)

            st.metric("R² Score", f"{r2:.3f}")
            st.metric("Mean Squared Error", f"{mse:.3f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred_lr, color='blue')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Linear Regression: Actual vs Predicted")
            st.pyplot(fig)

    # Random Forest
    with col2:
        st.subheader("🌲 Random Forest Regressor")
        if st.button("Run Random Forest"):
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            r2 = r2_score(y_test, y_pred_rf)
            mse = mean_squared_error(y_test, y_pred_rf)

            st.metric("R² Score", f"{r2:.3f}")
            st.metric("Mean Squared Error", f"{mse:.3f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred_rf, color='green')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Random Forest: Actual vs Predicted")
            st.pyplot(fig)