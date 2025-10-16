import streamlit as st

st.markdown("# User Registration")

with st.form("Form1"):
    first_name = st.text_input("First Name")
    last_name = st.text_input("Last Name")
    age = st.text_input("Age")
    submit_button = st.form_submit_button("Submit")

if submit_button:
    st.write(f"User Registered: {first_name} {last_name}, Age: {age}")
