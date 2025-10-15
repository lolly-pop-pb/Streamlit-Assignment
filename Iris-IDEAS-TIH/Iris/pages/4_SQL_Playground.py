import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import duckdb
from st_aggrid import AgGrid, GridOptionsBuilder

iris_table = load_iris()
df = pd.DataFrame(iris_table.data, columns=iris_table.feature_names)
df['target'] = iris_table.target

st.title("🦆 DuckDB Integration")

st.markdown("""
Use SQL queries to explore the Iris dataset via *DuckDB*.  
You can run custom SQL commands directly on the in-memory data.
""")

# Connect to DuckDB
conn = duckdb.connect(database=':memory:')
conn.register('iris_table', df)

# SQL query input
default_query = "SELECT * FROM iris_table LIMIT 5;"
query = st.text_area("💬 Enter your SQL query:", default_query, height=120)

if st.button("Run Query"):
    try:
        result_df = conn.execute(query).df()

        st.success("✅ Query executed successfully!")
        # --- Option 2: AgGrid Viewer ---
        st.subheader("🧩 AgGrid Data Explorer")
        gb = GridOptionsBuilder.from_dataframe(result_df)
        gb.configure_pagination(enabled=True)
        gb.configure_default_column(editable=False, groupable=True, sortable=True, filter=True, resizable=True)
        gb.configure_side_bar()
        grid_options = gb.build()

        AgGrid(
            result_df,
            gridOptions=grid_options,
            height=350,
            theme="alpine",  # Other options: "streamlit", "balham", "material"
            enable_enterprise_modules=False,
        )

        # --- Quick Visualization ---
        if not result_df.empty:
            numeric_cols = result_df.select_dtypes(include=['float', 'int']).columns
            if len(numeric_cols) >= 2:
                st.subheader("📈 Quick Visualization")
                x_axis = st.selectbox("Select X-axis", numeric_cols)
                y_axis = st.selectbox("Select Y-axis", numeric_cols, index=1)

                fig, ax = plt.subplots(figsize=(5, 4))
                sns.scatterplot(
                    data=result_df,
                    x=x_axis,
                    y=y_axis,
                    hue='target' if 'target' in result_df.columns else None,
                    ax=ax
                )
                ax.set_title("Scatter Plot of Query Results")
                st.pyplot(fig, use_container_width=False)

    except Exception as e:
        st.error(f"❌ Error executing query: {e}")