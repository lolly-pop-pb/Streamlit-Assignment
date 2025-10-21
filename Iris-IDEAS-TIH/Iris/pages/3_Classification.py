import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    return df, iris.target_names

@st.cache_resource
def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

df, target_names = load_iris_data()

X = df.drop(columns=['target', 'species'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title("🤖 Classification Models")

model_choice = st.radio("Select a model:", ["Logistic Regression", "Random Forest"])

if st.button("Run Model"):
    if model_choice == "Logistic Regression":
        st.subheader("📈 Logistic Regression")
        log_reg = train_logistic_regression(X_train, y_train)
        y_pred = log_reg.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.3f}")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True)).transpose())

        with st.expander("Show model details"):
            st.write("Model Parameters:", log_reg.get_params())
            st.write("Sample Predictions:", pd.DataFrame({
                "Actual": y_test[:5].values,
                "Predicted": y_pred[:5]
            }))

    elif model_choice == "Random Forest":
        st.subheader("🌲 Random Forest Classifier")
        rf = train_random_forest(X_train, y_train)
        y_pred = rf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.3f}")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True)).transpose())

        # Feature Importance Plot
        st.subheader("Feature Importance Plot")
        importances = rf.feature_importances_
        sorted_idx = np.argsort(importances)
        fig, ax = plt.subplots()
        ax.barh(np.array(X.columns)[sorted_idx], importances[sorted_idx])
        ax.set_title("Feature Importance (Random Forest)")
        st.pyplot(fig)

        with st.expander("Show model details"):
            st.write("Model Parameters:", rf.get_params())
            st.write("Sample Predictions:", pd.DataFrame({
                "Actual": y_test[:5].values,
                "Predicted": y_pred[:5]
            }))