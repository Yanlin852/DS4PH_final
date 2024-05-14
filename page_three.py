from utils import get_dataset, selected_dataset
import streamlit as st
import seaborn as sns
import pandas as pd
import time
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

start_time=time.time()
st.set_option('deprecation.showPyplotGlobalUse', False)
dataset_name=st.sidebar.selectbox("Select Dataset: ",('Heart Attack',"Breast Cancer","Diabetes"))
classifier_name = st.sidebar.selectbox("Select Classifier: ",("Logistic Regression","KNN","SVM","Decision Trees",
                                                              "Random Forest","Gradient Boosting","XGBoost"))
def selected_dataset(dataset_name):
    data = get_dataset(dataset_name)
    if dataset_name == "Heart Attack":
        X = data.drop(["output"], axis=1)
        Y = data.output
        return X, Y
    elif dataset_name == "Breast Cancer":
        X = data.drop(["id", "diagnosis"], axis=1)
        Y = data.diagnosis
        return X, Y
    elif dataset_name == "Diabetes":
        if "class" in data.columns:
            X = data.drop(["class"], axis=1)
            Y = data['class']
            return X, Y
        else:
            raise ValueError("Column 'class' not found in the dataset")

X, Y = selected_dataset(dataset_name)  
            
def add_parameter_ui(clf_name):
    params={}
    st.sidebar.write("Select values: ")

    if clf_name == "Logistic Regression":
        R = st.sidebar.slider("Regularization",0.1,10.0,step=0.1)
        MI = st.sidebar.slider("max_iter",50,400,step=50)
        params["R"] = R
        params["MI"] = MI

    elif clf_name == "KNN":
        K = st.sidebar.slider("n_neighbors",1,20)
        params["K"] = K

    elif clf_name == "SVM":
        C = st.sidebar.slider("Regularization",0.01,10.0,step=0.01)
        kernel = st.sidebar.selectbox("Kernel",("linear", "poly", "rbf", "sigmoid", "precomputed"))
        params["C"] = C
        params["kernel"] = kernel

    elif clf_name == "Decision Trees":
        M = st.sidebar.slider("max_depth", 2, 20)
        C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        SS = st.sidebar.slider("min_samples_split",1,10)
        params["M"] = M
        params["C"] = C
        params["SS"] = SS

    elif clf_name == "Random Forest":
        N = st.sidebar.slider("n_estimators",50,500,step=50,value=100)
        M = st.sidebar.slider("max_depth",2,20)
        C = st.sidebar.selectbox("Criterion",("gini","entropy"))
        params["N"] = N
        params["M"] = M
        params["C"] = C

    elif clf_name == "Gradient Boosting":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50,value=100)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5)
        L = st.sidebar.selectbox("Loss", ('deviance', 'exponential'))
        M = st.sidebar.slider("max_depth",2,20)
        params["N"] = N
        params["LR"] = LR
        params["L"] = L
        params["M"] = M

    elif clf_name == "XGBoost":
        N = st.sidebar.slider("n_estimators", 50, 500, step=50, value=50)
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5,value=0.1)
        O = st.sidebar.selectbox("Objective", ('binary:logistic','reg:logistic','reg:squarederror',"reg:gamma"))
        M = st.sidebar.slider("max_depth", 1, 20,value=6)
        G = st.sidebar.slider("Gamma",0,10,value=5)
        L = st.sidebar.slider("reg_lambda",1.0,5.0,step=0.1)
        A = st.sidebar.slider("reg_alpha",0.0,5.0,step=0.1)
        CS = st.sidebar.slider("colsample_bytree",0.5,1.0,step=0.1)
        params["N"] = N
        params["LR"] = LR
        params["O"] = O
        params["M"] = M
        params["G"] = G
        params["L"] = L
        params["A"] = A
        params["CS"] = CS

    RS=st.sidebar.slider("Random State",0,100)
    params["RS"] = RS
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):
    global clf
    if clf_name == "Logistic Regression":
        clf = LogisticRegression(C=params["R"],max_iter=params["MI"])

    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])

    elif clf_name == "SVM":
        clf = SVC(kernel=params["kernel"],C=params["C"])

    elif clf_name == "Decision Trees":
        clf = clf = DecisionTreeClassifier(max_depth=params["M"], criterion=params["C"])

    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(n_estimators=params["N"],max_depth=params["M"],criterion=params["C"])

    elif clf_name == "Gradient Boosting":
        clf = GradientBoostingClassifier(n_estimators=params["N"],learning_rate=params["LR"],loss=params["L"],max_depth=params["M"])

    elif clf_name == "XGBoost":
        clf = XGBClassifier(booster="gbtree",n_estimators=params["N"],max_depth=params["M"],learning_rate=params["LR"],
                            objective=params["O"],gamma=params["G"],reg_alpha=params["A"],reg_lambda=params["L"],colsample_bytree=params["CS"])

    return clf

clf = get_classifier(classifier_name,params)

#Build Model
def model():
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=65)

    #MinMax Scaling / Normalization of data
    Std_scaler = StandardScaler()
    X_train = Std_scaler.fit_transform(X_train)
    X_test = Std_scaler.transform(X_test)

    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test)
    acc=accuracy_score(Y_test,Y_pred)

    return Y_pred,Y_test

Y_pred,Y_test=model()

def compute(Y_pred,Y_test):
    #Plot PCA
    pca=PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:,0]
    x2 = X_projected[:,1]
    plt.figure(figsize=(16,8))
    plt.scatter(x1,x2,c=Y,alpha=0.8,cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar()
    st.pyplot()

    c1, c2 = st.columns((4,3))
    #Output plot
    plt.figure(figsize=(12,6))
    plt.scatter(range(len(Y_pred)),Y_pred,color="yellow",lw=5,label="Predictions")
    plt.scatter(range(len(Y_test)),Y_test,color="red",label="Actual")
    plt.title("Prediction Values vs Real Values")
    plt.legend()
    plt.grid(True)
    c1.pyplot()

    #Confusion Matrix
    cm=confusion_matrix(Y_test,Y_pred)
    class_label = ["High-risk", "Low-risk"]
    df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)
    plt.figure(figsize=(12, 7.5))
    sns.heatmap(df_cm,annot=True,cmap='Pastel1',linewidths=2,fmt='d')
    plt.title("Confusion Matrix",fontsize=15)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    c2.pyplot()

    #Calculate Metrics
    acc=accuracy_score(Y_test,Y_pred)
    mse=mean_squared_error(Y_test,Y_pred)
    precision = precision_score(Y_test, Y_pred, pos_label=1, average='binary')
    recall = recall_score(Y_test, Y_pred, pos_label=1, average='binary')
    fscore = f1_score(Y_test, Y_pred, pos_label=1, average='binary')
    st.subheader("Metrics of the model: ")
    st.text('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %\nMean Squared Error: {}'.format(
        round(precision, 3), round(recall, 3), round(fscore,3), round((acc*100),3), round((mse),3)))

st.markdown("<hr>",unsafe_allow_html=True)
st.header(f" Model for Prediction of {dataset_name}")
st.subheader(f"Classifier Used: {classifier_name}")
compute(Y_pred,Y_test)

#Execution Time

end_time=time.time()
st.info(f"Total execution time: {round((end_time - start_time),4)} seconds")

