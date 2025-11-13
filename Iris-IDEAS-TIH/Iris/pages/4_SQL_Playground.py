import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import duckdb
from st_aggrid import AgGrid, GridOptionsBuilder
import pdf_export

st.set_page_config(
    page_title="Iris DuckDB SQL Playground",
    page_icon="🦆",
    layout="wide"
)

# Inject custom CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
st.title("🦆 DuckDB SQL Playground")

st.markdown("""
Run SQL queries on the Iris dataset in-memory.  
You can use sample queries or write your own custom SQL.
""")

# Load Iris Dataset
@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].apply(lambda x: iris.target_names[x])
    return df

df = load_iris_data()

# DuckDB Connection
@st.cache_resource
def get_duckdb_connection(df):
    conn = duckdb.connect(database=":memory:")  # in-memory database
    conn.register("iris_table", df)
    return conn

conn = get_duckdb_connection(df)

# Sample Queries
sample_queries = {
    "Show first 5 rows": "SELECT * FROM iris_table LIMIT 5;",
    "Average of features by species": """
        SELECT species,
               AVG("sepal length (cm)") AS avg_sepal_length,
               AVG("sepal width (cm)") AS avg_sepal_width,
               AVG("petal length (cm)") AS avg_petal_length,
               AVG("petal width (cm)") AS avg_petal_width
        FROM iris_table
        GROUP BY species;
    """,
    "Count per species": "SELECT species, COUNT(*) AS count FROM iris_table GROUP BY species;"
}
# Session State Initialization
if "query_result" not in st.session_state:
    st.session_state.query_result = pd.DataFrame()
if "query_text" not in st.session_state:
    st.session_state.query_text = sample_queries["Show first 5 rows"]

# Query Input
selected_query = st.selectbox("Pick a sample query:", list(sample_queries.keys()))
st.session_state.query_text = st.text_area("Or write/edit your SQL query:", 
                                           value=sample_queries[selected_query], 
                                           height=120)
# Run Query Button
def run_query(sql):
    return conn.execute(sql).df()

if st.button("Run Query"):
    try:
        st.session_state.query_result = run_query(st.session_state.query_text)
        st.success("✅ Query executed successfully!")
    except Exception as e:
        st.session_state.query_result = pd.DataFrame()
        st.error(f"❌ Error executing query: {e}")

# Display AgGrid Table
if not st.session_state.query_result.empty:
    result_df = st.session_state.query_result
    st.subheader("🧩 AgGrid Data Explorer")
    gb = GridOptionsBuilder.from_dataframe(result_df)
    gb.configure_pagination(enabled=True)
    gb.configure_default_column(editable=False, groupable=True, sortable=True, filter=True, resizable=True)
    gb.configure_side_bar()
    gb.configure_grid_options(domLayout='autoHeight')
    grid_options = gb.build()
    AgGrid(
        result_df,
        gridOptions=grid_options,
        height=350,
        theme="alpine",
        enable_enterprise_modules=False,
    )
  
    # Quick Visualization (first two numeric columns)
    
    numeric_cols = result_df.select_dtypes(include=["float", "int"]).columns
    if len(numeric_cols) >= 2:
        x_axis, y_axis = numeric_cols[:2]
        st.subheader("📈 Quick Visualization")
        st.markdown(f"Scatter plot for **{x_axis}** vs **{y_axis}**")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=result_df,
            x=x_axis,
            y=y_axis,
            hue="species" if "species" in result_df.columns else None,
            ax=ax
        )
        ax.set_title(f"{x_axis} vs {y_axis}")
        st.pyplot(fig)

# --- PDF Export Section ---
st.markdown("---")
st.subheader("📄 Export Query Results & Visuals")
st.markdown("You can export the **current query results and the plot** as a single PDF file.")

if st.button("📤 Export Full Page to PDF"):
    try:
        figs_to_export = []

        # 1️⃣ SQL Query Description as figure
        fig_desc, ax_desc = plt.subplots(figsize=(8, 3))
        ax_desc.axis("off")
        query_text = st.session_state.query_text or "No query executed yet."
        ax_desc.text(0, 0.5, f"🦆 SQL Query Executed:\n\n{query_text}", fontsize=10, va='top')
        figs_to_export.append(fig_desc)

        # 2️⃣ Query Result Table as figure (show up to 15 rows)
        if not st.session_state.query_result.empty:
            fig_table, ax_table = plt.subplots(figsize=(10, 6))
            ax_table.axis("off")
            display_df = st.session_state.query_result.head(15)
            ax_table.table(
                cellText=display_df.values,
                colLabels=display_df.columns,
                loc='center',
                cellLoc='center'
            )
            ax_table.set_title("📄 Query Result Preview (first 15 rows)")
            figs_to_export.append(fig_table)

        # 3️⃣ Quick Visualization (if exists)
        numeric_cols = st.session_state.query_result.select_dtypes(include=["float", "int"]).columns
        if len(numeric_cols) >= 2:
            x_axis, y_axis = numeric_cols[:2]
            fig_plot, ax_plot = plt.subplots(figsize=(6, 4))
            import seaborn as sns
            sns.scatterplot(
                data=st.session_state.query_result,
                x=x_axis,
                y=y_axis,
                hue="species" if "species" in st.session_state.query_result.columns else None,
                ax=ax_plot
            )
            ax_plot.set_title(f"📈 {x_axis} vs {y_axis}")
            figs_to_export.append(fig_plot)

        if figs_to_export:
            pdf_path = pdf_export.create_multi_fig_pdf(
                figs_to_export, title="DuckDB SQL Playground - Full Page Export"
            )
            st.success(f"✅ PDF exported successfully! Saved to: {pdf_path}")
            st.info("You can find it inside the exports folder.")
        else:
            st.warning("⚠️ No data to export!")

    except Exception as e:
        st.error(f"❌ Export failed: {e}")