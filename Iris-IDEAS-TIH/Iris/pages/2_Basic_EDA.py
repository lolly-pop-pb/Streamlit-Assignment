import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.title("📊 Exploratory Data Analysis")
st.markdown("Here we explore the distribution and relationships among the Iris dataset features.")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
target_names = iris.target_names

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

features = iris.feature_names
st.subheader("Feature Histograms")
fig, axes = plt.subplots(2, 2, figsize=(6, 4))
for i, feature in enumerate(features):
    row, col = divmod(i, 2)
    sns.histplot(df[feature], kde=True, color='teal', ax=axes[row, col])
plt.tight_layout()
st.pyplot(fig, use_container_width=False)

st.subheader("Mean of Each Feature per Species (Line Chart)")
mean_df = df.groupby("target")[iris.feature_names].mean().T
mean_df.columns = target_names
fig, ax = plt.subplots(figsize=(6, 4))
mean_df.plot(kind="line", ax=ax, marker='o')
ax.set_title("Feature Means by Species")
ax.set_ylabel("Mean Value (cm)")
st.pyplot(fig)

st.subheader("Boxplot: Feature Distributions by Species")
feature = st.selectbox("Select a feature for boxplot:", iris.feature_names)
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(x="target", y=feature, data=df, palette="Set2", ax=ax)
ax.set_xticklabels(target_names)
ax.set_title(f"{feature} by Species")
st.pyplot(fig)

st.subheader("Violin Plot: Detailed Feature Distribution")
feature_v = st.selectbox("Select a feature for violin plot:", iris.feature_names, key="violin")
fig, ax = plt.subplots(figsize=(6, 4))
sns.violinplot(x="target", y=feature_v, data=df, palette="coolwarm", ax=ax)
ax.set_xticklabels(target_names)
ax.set_title(f"{feature_v} by Species")
st.pyplot(fig)