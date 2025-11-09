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
import pdf_export  

# Load Iris Data

@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = df['target'].apply(lambda x: iris.target_names[x])
    return df, iris.target_names

df, target_names = load_iris_data()
X = df.drop(columns=['target', 'species'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train Models (Cached)

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

st.title("🤖 Iris Classification Models")


# Model Selection

model_choice = st.radio("Select a model:", ["Logistic Regression", "Random Forest"])

# Initialize session_state
if "export_figs" not in st.session_state:
    st.session_state.export_figs = []
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "export_title" not in st.session_state:
    st.session_state.export_title = ""

# Run Model
if st.button("Run Model"):
    st.session_state.export_figs = []  # reset previous figures
    st.session_state.export_title = f"{model_choice} Results"

    if model_choice == "Logistic Regression":
        st.subheader("📈 Logistic Regression")
        model = train_logistic_regression(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.3f}")

        # Confusion matrix
        fig_cm, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_title("Confusion Matrix - Logistic Regression")
        st.pyplot(fig_cm)
        st.session_state.export_figs.append(fig_cm)

        # Classification report
        report_df = pd.DataFrame(classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        )).transpose()
        st.subheader("Classification Report")
        st.dataframe(report_df)

        # Report as matplotlib table
        fig_report, ax_report = plt.subplots(figsize=(8, 4))
        ax_report.axis("off")
        ax_report.table(cellText=report_df.round(3).values,
                        colLabels=report_df.columns,
                        rowLabels=report_df.index,
                        loc='center')
        ax_report.set_title("Classification Report Table")
        st.session_state.export_figs.append(fig_report)

        # Model details
        with st.expander("Show model details"):
            st.write("Model Parameters:", model.get_params())

    elif model_choice == "Random Forest":
        st.subheader("🌲 Random Forest Classifier")
        model = train_random_forest(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.3f}")

        # Confusion matrix
        fig_cm, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens',
                    xticklabels=target_names, yticklabels=target_names, ax=ax)
        ax.set_title("Confusion Matrix - Random Forest")
        st.pyplot(fig_cm)
        st.session_state.export_figs.append(fig_cm)

        # Classification report
        report_df = pd.DataFrame(classification_report(
            y_test, y_pred, target_names=target_names, output_dict=True
        )).transpose()
        st.subheader("Classification Report")
        st.dataframe(report_df)

        # Report as matplotlib table
        fig_report, ax_report = plt.subplots(figsize=(8, 4))
        ax_report.axis("off")
        ax_report.table(cellText=report_df.round(3).values,
                        colLabels=report_df.columns,
                        rowLabels=report_df.index,
                        loc='center')
        ax_report.set_title("Classification Report Table")
        st.session_state.export_figs.append(fig_report)

        # Feature importance
        st.subheader("Feature Importance Plot")
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        fig_imp, ax_imp = plt.subplots()
        ax_imp.barh(np.array(X.columns)[sorted_idx], importances[sorted_idx])
        ax_imp.set_title("Feature Importance (Random Forest)")
        st.pyplot(fig_imp)
        st.session_state.export_figs.append(fig_imp)

    # Mark that model has been trained
    st.session_state.model_trained = True

# Export PDF - only show if model has been trained
if st.session_state.model_trained:
    st.markdown("---")
    st.subheader("📄 Export Model Results")

    if st.button("📤 Export Model Results to PDF"):
        try:
            pdf_path = pdf_export.create_multi_fig_pdf(
                st.session_state.export_figs,
                title=st.session_state.export_title
            )
            st.success(f"✅ PDF exported successfully! Saved to: {pdf_path}")
            st.info("You can find it inside the Iris/exports folder.")
        except Exception as e:
            st.error(f"❌ Export failed: {e}")