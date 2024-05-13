from pathlib import Path

import streamlit as st

with st.echo("below"):
    from st_pages import Page, add_page_title, show_pages

    "## Declaring the pages in your app:"

    show_pages(
        [
            Page("streamlit_app.py", "Home", "🏠"),
            # Can use :<icon-name>: or the actual icon
            Page("page_one.py", "Our Data", ":books:"),
            Page("page_two.py", "Data Analysis", "📊"),
            Page("page_three.py", "Classifer", "🔧"),
            Page("page_four.py", "Prediction", "🩺"),
        ]
    )

    add_page_title()  # Optional method to add title and icon to current page

"## Alternative approach, using a config file"

"Contents of `.streamlit/pages.toml`"

st.code(Path(".streamlit/pages.toml").read_text(), language="toml")

"Streamlit script:"

with st.echo("below"):
    from st_pages import show_pages_from_config

    show_pages_from_config()


with st.expander("Show documentation"):
    st.help(show_pages)

    st.help(Page)

    st.help(add_page_title)