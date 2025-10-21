import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import duckdb
from st_aggrid import AgGrid, GridOptionsBuilder

st.title("🦆 DuckDB Integration")

st.markdown("""
Use SQL queries to explore the Iris dataset via *DuckDB*.  
You can run custom SQL commands directly on the in-memory data.
""")

@st.cache_data
def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["species"] = df["target"].apply(lambda x: iris.target_names[x])
    return df

df = load_iris_data()

# Connect to DuckDB
@st.cache_resource
def get_duckdb_connection(df):
    conn = duckdb.connect(database="iris_data.duckdb", read_only=False)
    conn.register("iris_table", df)
    return conn

conn = get_duckdb_connection(df)

st.subheader("💬 SQL Playground")

# Sample helper queries
sample_queries = {
    "Show first 5 rows": "SELECT * FROM iris_table LIMIT 5;",
    "Average of features by species": """
        SELECT species,
               AVG("sepal length (cm)") AS avg_sepal_length,
               AVG("petal length (cm)") AS avg_petal_length
        FROM iris_table
        GROUP BY species;
    """,
    "Count per species": "SELECT species, COUNT(*) AS count FROM iris_table GROUP BY species;"
}

selected_query = st.selectbox("Pick a sample query:", list(sample_queries.keys()))
default_query = sample_queries[selected_query]
# Query input box
query = st.text_area("Or write/edit your SQL query:", default_query, height=120)

if st.button("Run Query"):
    try:
        @st.cache_data(show_spinner=False)
        def run_query(sql):
            return conn.execute(sql).df()

        result_df = run_query(query)

        st.success("✅ Query executed successfully!")

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

        if not result_df.empty:
            numeric_cols = result_df.select_dtypes(include=["float", "int"]).columns
            if len(numeric_cols) >= 2:
                st.subheader("📈 Quick Visualization")

                x_axis, y_axis = numeric_cols[:2]  # first two numeric columns
                st.markdown(f"Scatter plot for **{x_axis}** vs **{y_axis}**")

                fig, ax = plt.subplots(figsize=(5, 4))
                sns.scatterplot(
                    data=result_df,
                    x=x_axis,
                    y=y_axis,
                    hue="species" if "species" in result_df.columns else None,
                    ax=ax
                )
                ax.set_title(f"{x_axis} vs {y_axis}")
                st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error executing query: {e}")