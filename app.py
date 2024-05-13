from pathlib import Path
import streamlit as st
from st_pages import Page, add_page_title, show_pages

show_pages(
        [
            Page("app.py", "Home", "🏠"),
            # Can use :<icon-name>: or the actual icon
            Page("page_one.py", "Our Data", ":books:"),
            Page("page_two.py", "Data Analysis", "📊"),
            Page("page_three.py", "Classifer", "🔧"),
            Page("page_four.py", "Prediction", "🩺"),
        ]
    )

st.title("Welcome to the Healthcare Analysis APP")

st.header("People")
st.markdown('👩🏻‍💻Yiyang You&nbsp;&nbsp;&nbsp; 👩🏻‍💻Kaitlin Zhu&nbsp;&nbsp;&nbsp;👩🏻‍💻Yanlin Wu')
st.header("Introduction")
st.markdown("""
Welcome to our interactive web app designed to provide comprehensive insights into healthcare data. We offers tools for data visualization, regression, and predictive analytics. Focused on critical health issues, our app provides detailed analysis for
- Heart Attact
- Breast Cancer
- Diabetes
""")

st.markdown("This app is structured into four main sections, each tailored to offer specific functionalities:")


st.subheader("Our Data")
st.markdown("""
In the 'Our Data' section, users can explore the raw healthcare data that forms the basis of our analyses. This page provides a clear view of the dataset attributes such as age, sex, heart rate, blood sugar levels, and more.
""")

st.subheader("Data Analysis")
st.markdown("""
The 'Data Analysis' page delves deeper into the dataset, offering visualizations and statistical analyses that highlight key trends and patterns. Explore various charts and graphs to understand the relationships between different health indicators.
""")

st.subheader("Classifier")
st.markdown("""
On the 'Classifier' page, users can select from a range of machine learning classifiers to perform regression analysis on the data. This section allows for the application of different algorithms to see how they predict health outcomes based on the input features.
""")

st.subheader("Prediction")
st.markdown("""
The 'Prediction' section enables users to input their personal health metrics to receive individual risk assessments for conditions like heart attacks, breast cancer, and diabetes. It provides personalized predictions based on the selected classifier and user-provided data.
""")

st.markdown("""
Navigate through the app using the sidebar to select different pages and utilize the features each one offers. Enjoy your exploration and insights discovery!
""")




