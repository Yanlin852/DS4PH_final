import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st

LE = LabelEncoder()

def get_dataset(dataset_name):
    if dataset_name == "Heart Attack":
        data = pd.read_csv("https://raw.githubusercontent.com/Yanlin852/DS4PH_final/main/heart.csv")
        # data = pd.read_csv("/Users/1rin/Desktop/JHU/24spring/2_DS4PH/DS4PH_final/heart.csv")
        return data
    elif dataset_name == "Breast Cancer":
        data = pd.read_csv("https://raw.githubusercontent.com/Yanlin852/DS4PH_final/main/BreastCancer.csv")
        # data = pd.read_csv("/Users/1rin/Desktop/JHU/24spring/2_DS4PH/DS4PH_final/BreastCancer.csv")
        data["diagnosis"] = LE.fit_transform(data["diagnosis"])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data["diagnosis"] = pd.to_numeric(data["diagnosis"], errors="coerce")
        return data
    elif dataset_name == "Diabetes":
        data = pd.read_csv("https://raw.githubusercontent.com/KaitlinZhu/capstone/main/diabetes_data_upload.csv")
        data["class"] = LE.fit_transform(data["class"])
        data["class"] = pd.to_numeric(data["class"], errors="coerce")
        return data

def selected_dataset(dataset_name):
    data = get_dataset(dataset_name)
    if dataset_name == "Heart Attack":
        X = data.drop(["output"], axis=1)
        Y = data.output
        age = data.age
        sex = data.sex
        return X, Y, age, sex
    elif dataset_name == "Breast Cancer":
        X = data.drop(["id", "diagnosis"], axis=1)
        Y = data.diagnosis
        return X, Y
    elif dataset_name == "Diabetes":
        if "class" in data.columns:
            X = data.drop(["class"], axis=1)
            Y = data['class']
            age = data.Age
            gender = data.Gender
            return X, Y, age, gender
        else:
            raise ValueError("Column 'class' not found in the dataset")