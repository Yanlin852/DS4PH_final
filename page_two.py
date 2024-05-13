from utils import get_dataset, selected_dataset
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


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

st.set_option('deprecation.showPyplotGlobalUse', False)

def plot_op(dataset_name):
    col1, col2 = st.columns((1, 5))
    plt.figure(figsize=(12, 8))
    plt.title(f"Classes in {dataset_name}")
    if dataset_name == "Heart Attack":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        col2.pyplot()

    elif dataset_name == "Breast Cancer":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        col2.pyplot()
        
    elif dataset_name == "Diabetes":
        col1.write(Y)
        sns.countplot(Y, palette='gist_heat')
        col2.pyplot()

st.write("Shape of dataset: ",data.shape)
st.write("Number of classes: ",Y.nunique())
plot_op(dataset_name)