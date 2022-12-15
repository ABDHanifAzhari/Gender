import pandas as pd
import streamlit as st
import numpy as np
from numpy import array
from sklearn.utils.validation import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
import altair as alt

# create content
st.sidebar.title("Menu")
menu = st.sidebar.radio ("",["Home","Preprocessing","Modelling & Evaluations","Implementasi"])
df = pd.read_csv("https://raw.githubusercontent.com/ABDHanifAzhari/dataset/main/gender_classification_v7%20(1).csv")

if menu == "Home" :
    st.title("Prediksi Jenis Kelamin")
    st.container()
    st.write("Website ini bertujuan untuk memprediksi jenis kelamin dari seseorang, dengan menggunakan data yang terdapat pada dataset yang telah ada")
    st.header("Sampel Data")
    # read data
    st.text("""
    kolom yang digunakan:
    * long_hair                   : rambut panjang (1/0)
    * forehead_width_cm           : lebar dahi dalam cm
    * forehead_heigh_cm           : tinggi dahi dalam cm
    * nose_wide                   : hidung lebar (1/0)
    * nose_long                   : hidung panjang (1/0)
    * lips_thin                   : bibir tipis (1/0)
    * distance_nose_to_lip_long   : jarak hidung dengan bibir (1/0)
    """)
    st.caption('link datasets : https://www.kaggle.com/datasets/elakiricoder/gender-classification-dataset')
    st.dataframe(df)
    row, col = df.shape
    st.caption(f"({row} rows, {col} cols)")

elif menu == "Preprocessing":
    # section output
    st.header("Preprocessing")
    # data test dan data uji
    X = df.drop(columns="gender")
    y = df.gender
    # 
    judul = X.columns.copy()
    # mengjitung hasil normalisasi
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    hasil = pd.DataFrame(X,columns=judul)
    st.dataframe(hasil)

elif menu == "Modelling & Evaluations":
    # section output
    st.header("Model")
    # data test dan data uji
    X = df.drop(columns="gender")
    y = df.gender
    # 
    judul = X.columns.copy()
    # mengjitung hasil normalisasi
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    hasil = pd.DataFrame(X,columns=judul)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    metode1 = KNeighborsClassifier(n_neighbors=3)
    metode1.fit(X_train, y_train)

    metode2 = GaussianNB()
    metode2.fit(X_train, y_train)

    metode3 = tree.DecisionTreeClassifier(criterion="gini")
    metode3.fit(X_train, y_train)

    st.write ("Pilih metode yang ingin anda gunakan :")
    met1 = st.checkbox("KNN")
    if met1 :
        st.write("Hasil Akurasi Data Training Menggunakan KNN sebesar : ", (100 * metode1.score(X_train, y_train)))
        st.write("Hasil Akurasi Data Testming Menggunakan KNN sebesar : ", (100 * metode1.score(X_test, y_test)))
    met2 = st.checkbox("Naive Bayes")
    if met2 :
        st.write("Hasil Akurasi Data Training Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_train, y_train)))
        st.write("Hasil Akurasi Data Testing Menggunakan Naive Bayes sebesar : ", (100 * metode2.score(X_test, y_test)))
    met3 = st.checkbox("Decesion Tree")
    if met3 :
        st.write("Hasil Akurasi Data Training Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_train, y_train)))
        st.write("Hasil Akurasi Data Testing Menggunakan Decission Tree sebesar : ", (100 * metode3.score(X_test, y_test)))
    
    submit2 = st.button("Submit")
    if submit2:      
        if met1 :
            st.write("Metode yang Anda gunakan Adalah KNN")

        elif met2 :
            st.write("Metode yang Anda gunakan Adalah Naive Bayes")

        elif met3 :
            st.write("Metode yang Anda gunakan Adalah Decesion Tree")

        else :
            st.write("Anda Belum Memilih Metode")
    
    

elif menu == "Implementasi":
    st.header("Hasil Predict Jenis Kelamin")
    st.subheader("Input Data")
    # create input
    long_hair = st.number_input("long hair ", 0, 1, step=1)
    forehead_width = st.number_input("forehead width (cm)", 10.0, 16.0, step=0.1)
    forehead_height = st.number_input("forehead height (cm)", 5.0, 7.5, step=0.1)
    nose_wide = st.number_input("nose wide", 0, 1, step=1)
    nose_long = st.number_input("nose long", 0, 1, step=1)
    lips_thin = st.number_input("lips thin", 0, 1, step=1)
    distance_nose_to_lip_long = st.number_input("distance nose to lip", 0, 1, step=1)

    # create button submit
    def submit():
        # input
        inputs = np.array([[long_hair, forehead_width, forehead_height, nose_wide, nose_long, lips_thin, distance_nose_to_lip_long]])
        st.write(inputs)

        # import label encoder
        le = joblib.load("le.save")

        # create 3 output
        col1, col2, col3 = st.columns(3)
        with col1:
            model1 = joblib.load("nb.joblib")
            y_pred1 = model1.predict(inputs)
            col1.subheader("Gaussian Naive Bayes")
            col1.write(f"Result : {le.inverse_transform(y_pred1)[0]}")


        with col2:
            model2 = joblib.load("knn.joblib")
            y_pred2 = model2.predict(inputs)
            st.subheader("k-nearest neighbors")
            col2.write(f"Result : {le.inverse_transform(y_pred2)[0]}")

        with col3:
            model3 = joblib.load("tree.joblib")
            y_pred3 = model3.predict(inputs)
            st.subheader("Decision Tree")
            col3.write(f"Result : {le.inverse_transform(y_pred3)[0]}")

    # section output
    submitted = st.button("Submit")
    if submitted:
        submit()
