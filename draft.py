import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import duckdb

# Load dataset
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['target'] = iris_data.target
target_names = iris_data.target_names

st.set_page_config(page_title="Iris Dataset App")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📘 Dataset Overview", "📊 Exploratory Data Analysis", "🤖 ML Models", "🦆 DuckDB SQL Playground"])

# TAB 1: Overview
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

# TAB 2: EDA
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

    features = iris_data.feature_names
    st.subheader("Feature Histograms")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i, feature in enumerate(features):
        row, col = divmod(i, 2)
        sns.histplot(df[feature], kde=True, color='teal', ax=axes[row, col])
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)\
    
    st.subheader("Mean of Each Feature per Species (Line Chart)")
    mean_df = df.groupby("target")[iris_data.feature_names].mean().T
    mean_df.columns = target_names
    fig, ax = plt.subplots(figsize=(6, 4))
    mean_df.plot(kind="line", ax=ax, marker='o')
    ax.set_title("Feature Means by Species")
    ax.set_ylabel("Mean Value (cm)")
    st.pyplot(fig)

    st.subheader("Boxplot: Feature Distributions by Species")
    feature = st.selectbox("Select a feature for boxplot:", iris_data.feature_names)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="target", y=feature, data=df, palette="Set2", ax=ax)
    ax.set_xticklabels(target_names)
    ax.set_title(f"{feature} by Species")
    st.pyplot(fig)

    st.subheader("Violin Plot: Detailed Feature Distribution")
    feature_v = st.selectbox("Select a feature for violin plot:", iris_data.feature_names, key="violin")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(x="target", y=feature_v, data=df, palette="coolwarm", ax=ax)
    ax.set_xticklabels(target_names)
    ax.set_title(f"{feature_v} by Species")
    st.pyplot(fig)


# TAB 3: Machine Learning Models
with tab3:
    st.title("🤖 Classification Models")
    st.write("Predict the **species** of Iris flowers based on their features using classification algorithms.")

    # Features and target
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    col1, col2 = st.columns(2)

    # Logistic Regression
    with col1:
        st.subheader("📈 Logistic Regression")
        if st.button("Run Logistic Regression"):
            log_reg = LogisticRegression(max_iter=200)
            log_reg.fit(X_train, y_train)
            y_pred_lr = log_reg.predict(X_test)

            acc = accuracy_score(y_test, y_pred_lr)
            cm = confusion_matrix(y_test, y_pred_lr)
            report = classification_report(y_test, y_pred_lr, target_names=target_names, output_dict=True)

            st.metric("Accuracy", f"{acc:.3f}")

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=target_names, yticklabels=target_names, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

    # Random Forest Classifier
    with col2:
        st.subheader("🌲 Random Forest Classifier")
        if st.button("Run Random Forest Classifier"):
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)

            acc = accuracy_score(y_test, y_pred_rf)
            cm = confusion_matrix(y_test, y_pred_rf)
            report = classification_report(y_test, y_pred_rf, target_names=target_names, output_dict=True)

            st.metric("Accuracy", f"{acc:.3f}")

            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                        xticklabels=target_names, yticklabels=target_names, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(report).transpose())

# TAB 4: DuckDB SQL Playground
with tab4:
    st.title("🦆 DuckDB Integration")

    st.markdown("""
    Use SQL queries to explore the Iris dataset via **DuckDB**.
    You can run custom SQL commands directly on the in-memory data.
    """)

    # Connect to DuckDB
    conn = duckdb.connect(database=':memory:')
    conn.register('iris_table', df)

    # Text area for SQL input
    default_query = "SELECT * FROM iris_table LIMIT 5;"
    query = st.text_area("Enter your SQL query:", default_query, height=120)

    if st.button("Run Query"):
        try:
            result_df = conn.execute(query).df()
            st.success("✅ Query executed successfully!")
            st.dataframe(result_df, use_container_width=True)
            # Optional visualization for numeric results
            if not result_df.empty:
                numeric_cols = result_df.select_dtypes(include=['float', 'int']).columns
                if len(numeric_cols) >= 2:
                    st.subheader("📈 Quick Visualization")
                    x_axis = st.selectbox("X-axis", numeric_cols)
                    y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.scatterplot(data=result_df, x=x_axis, y=y_axis, ax=ax, hue=result_df.columns[0] if 'target' in result_df.columns else None)
                    st.pyplot(fig, use_container_width=False)
        except Exception as e:
            st.error(f"❌ Error: {e}")