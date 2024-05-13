from utils import get_dataset, selected_dataset
import streamlit as st

dataset_name = st.selectbox(
        'Choose a disease to display:',
        ('Heart Attack', 'Breast Cancer', 'Diabetes') 
    )

data = get_dataset(dataset_name)
X, Y = selected_dataset(dataset_name)

st.write(data)
