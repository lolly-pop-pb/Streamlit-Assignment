import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.title("🤖 Classification Models")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

col1, col2 = st.columns(2)

# Logistic Regression
with col1:
    st.subheader("📈 Logistic Regression")
    if st.button("Run Logistic Regression"):
        log_reg = LogisticRegression(max_iter=200)
        log_reg.fit(X_train, y_train)
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

# Random Forest Classifier
with col2:
    st.subheader("🌲 Random Forest Classifier")
    if st.button("Run Random Forest Classifier"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
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
