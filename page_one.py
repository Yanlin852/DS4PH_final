from utils import get_dataset, selected_dataset
import streamlit as st

dataset_name = st.selectbox(
        'Choose a disease to display:',
        ('Heart Attack', 'Breast Cancer', 'Diabetes') 
    )

data = get_dataset(dataset_name)
if dataset_name == "Heart Attack":
    X, Y, age, sex = selected_dataset(dataset_name)
elif dataset_name == "Breast Cancer":
    X ,Y = selected_dataset(dataset_name)
elif dataset_name == "Diabetes":
    X, Y, age, gender = selected_dataset(dataset_name)
st.write(data)
