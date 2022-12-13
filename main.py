
import streamlit as st

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets
from sklearn.preprocessing import MinMaxScaler

import pickle

from sklearn import metrics


st.title('Prediksi Penyakit Cacar Monyet')
st.write("""
Aplikasi untuk meprediksi seseorang positif atau negatif cacar monyet
""")
tab1, tab2, tab3, tab4= st.tabs(["Data Understanding", "Preprocessing", "Modeling", "Implementation"])
# create content
with tab1:
    df = pd.read_csv("monkey_pox.csv")
    df2 = pd.read_csv("monkey_pox.csv")
    st.write("""
    <h5>Data Understanding</h5>
    """, unsafe_allow_html=True)
    st.markdown("""
    Link Repository Github
    <a href="https://github.com/FajarFatha/monkey-pox-prediction">https://github.com/FajarFatha/monkey-pox-prediction</a>
    """, unsafe_allow_html=True)

    st.markdown("""
    Link Dataset
    <a href="https://www.kaggle.com/datasets/muhammad4hmed/monkeypox-patients-dataset"> https://www.kaggle.com/datasets/muhammad4hmed/monkeypox-patients-dataset</a>
    """, unsafe_allow_html=True)

    st.write(df2)
    
    st.write("Penjelasan Fitur yang Ada : ")
    st.write("""
    <ol>
    <li>Patient_ID : Id dari pasien</li>
    <li>Systemic Illness : yaitu sebagai kelainan yang dapat mempengaruhi beberapa organ dan jaringan atau bahkan seluruh tubuh. (Fever = Demam, Swollen Lymph Nodes = Pembengkakan Kelenjar Getah Bening, None = Tidak Ada, Muscle Aches and Pain = Nyeri dan Nyeri Otot)</li>
    <li>Rectal Pain : nyeri rektum, rektum adalah usus besar bagian paling ujung</li>
    <li>Sore Throat : Sakit Tenggorokan</li>
    <li>Penile Oedema : Pembengkakan penis yang tidak disertai nyeri</li>
    <li>Oral Lesions : ulkus yang terjadi pada selaput lendir rongga mulut</li>
    <li>Solitary Lesion : Lesi soliter</li>
    <li>Swollen Tonsils : bengkak tonsil, Tonsil adalah kelenjar getah bening di bagian belakang mulut dan tenggorok bagian atas</li>
    <li>HIV Infection: Infeksi HIV</li>
    <li>Sexually Transmitted Infection : Penyakit menular seksual lain</li>
    <li>MonkeyPox : hasil prediksi, positif atau negatif</li>
    </ol>
    """,unsafe_allow_html=True)

with tab2:
    st.write("""
    <h5>Preprocessing</h5>
    """, unsafe_allow_html=True)
    st.write("""
    <p style="text-align: justify;text-indent: 45px;">Preprocessing data adalah proses mengubah data mentah ke dalam bentuk yang lebih mudah dipahami. Proses ini diperlukan untuk memperbaiki kesalahan pada data mentah yang seringkali tidak lengkap dan memiliki format yang tidak teratur. Preprocessing melibatkan proses validasi dan imputasi data.</p>
    <p style="text-align: justify;text-indent: 45px;">Salah satu tahap Preprocessing data adalah Normalisasi. Normalisasi data adalah elemen dasar data mining untuk memastikan record pada dataset tetap konsisten. Dalam proses normalisasi diperlukan transformasi data atau mengubah data asli menjadi format yang memungkinkan pemrosesan data yang efisien.</p>
    <br>
    """,unsafe_allow_html=True)
    st.container()
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    df['Systemic Illness'] = labelencoder.fit_transform(df['Systemic Illness'])
    df['Rectal Pain'] = labelencoder.fit_transform(df['Rectal Pain'])
    df['Sore Throat'] = labelencoder.fit_transform(df['Sore Throat'])
    df['Penile Oedema'] = labelencoder.fit_transform(df['Penile Oedema'])
    df['Oral Lesions'] = labelencoder.fit_transform(df['Oral Lesions'])
    df['Solitary Lesion'] = labelencoder.fit_transform(df['Solitary Lesion'])
    df['Swollen Tonsils'] = labelencoder.fit_transform(df['Swollen Tonsils'])
    df['HIV Infection'] = labelencoder.fit_transform(df['HIV Infection'])
    df['Sexually Transmitted Infection'] = labelencoder.fit_transform(df['Sexually Transmitted Infection'])
    df['MonkeyPox'] = labelencoder.fit_transform(df['MonkeyPox'])
    scaler = st.radio(
    "Lihat data sebelum dan setelah di preprocessing",
    ('sebelum', 'setelah'))
    if scaler == 'sebelum':
        st.write("Dataset Sebelum dipreprocessing : ")
        st.write(df2)
    elif scaler == 'setelah':
        st.write("Dataset Setelah dipreprocessing : ")
        st.write(df)

with tab3:
    st.write("""
    <h5>Modelling</h5>
    """, unsafe_allow_html=True)
    st.container()
    X=df.iloc[:,1:10].values
    y=df.iloc[:,10].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)
    algoritma = st.radio(
    "Pilih algoritma klasifikasi",
    ('KNN','Naive Bayes','Random Forest'))
    if algoritma=='KNN':
        model = KNeighborsClassifier(n_neighbors=3)
        filename='knn.pkl'
    elif algoritma=='Naive Bayes':
        model = GaussianNB()
        filename='naivebayes.pkl'
    elif algoritma=='Random Forest':
        model = RandomForestClassifier(n_estimators = 100)
        filename='randomforest.pkl'
    elif algoritma=='Ensemble Stacking':
        estimators = [
            ('rf_1', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('knn_1', KNeighborsClassifier(n_neighbors=10))             
        ]
        model = StackingClassifier(estimators=estimators, final_estimator=GaussianNB())
        filename='stacking.pkl'
    model.fit(X_train, y_train)
    Y_pred = model.predict(X_test) 
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,Y_pred)
    conf_matrix = pd.DataFrame(data=cm, columns=['Positif:1', 'Negatif:0'], index=['Positif:1','Negatif:0'])
    import matplotlib.pyplot as plt
    plt.figure()
    import seaborn as sns 
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
    score=metrics.accuracy_score(y_test,Y_pred)

    loaded_model = pickle.load(open(filename, 'rb'))
    st.write(f"akurasi : {score*100} %")
    st.write("Confusion Metrics")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

with tab4:
    st.write("""
    <h5>Implementation</h5>
    """, unsafe_allow_html=True)
    Systemic_Illness=st.selectbox(
        'Systemic Illness',
        ('Fever', 'Swollen Lymph Nodes', 'None', 'Muscle Aches and Pain')
    )
    if Systemic_Illness=='Fever':
        Systemic_Illness=0
    elif Systemic_Illness=='Swollen Lymph Nodes':
        Systemic_Illness=3
    elif Systemic_Illness=='None':
        Systemic_Illness=2
    elif Systemic_Illness=='Muscle Aches and Pain':
        Systemic_Illness=1

    Rectal_Pain=st.selectbox(
        'Rectal Pain',
        ('Ya','Tidak')
    )
    if Rectal_Pain=='Ya':
        Rectal_Pain=1
    elif Rectal_Pain=='Tidak':
        Rectal_Pain=0

    Sore_Throat=st.selectbox(
        'Sore Throat',
        ('Ya','Tidak')
    )
    if Sore_Throat=='Ya':
        Sore_Throat=1
    elif Sore_Throat=='Tidak':
        Sore_Throat=0

    Penile_Oedema=st.selectbox(
        'Penile Oedema',
        ('Ya','Tidak')
    )
    if Penile_Oedema=='Ya':
        Penile_Oedema=1
    elif Penile_Oedema=='Tidak':
        Penile_Oedema=0
    
    Oral_Lesions=st.selectbox(
        'Oral Lesions',
        ('Ya','Tidak')
    )
    if Oral_Lesions=='Ya':
        Oral_Lesions=1
    elif Oral_Lesions=='Tidak':
        Oral_Lesions=0

    Solitary_Lesion=st.selectbox(
        'Solitary Lesion',
        ('Ya','Tidak')
    )
    if Solitary_Lesion=='Ya':
        Solitary_Lesion=1
    elif Solitary_Lesion=='Tidak':
        Solitary_Lesion=0

    Swollen_Tonsils=st.selectbox(
        'Swollen Tonsils',
        ('Ya','Tidak')
    )
    if Swollen_Tonsils=='Ya':
        Swollen_Tonsils=1
    elif Swollen_Tonsils=='Tidak':
        Swollen_Tonsils=0

    HIV_Infection=st.selectbox(
        'HIV Infection',
        ('Ya','Tidak')
    )
    if HIV_Infection=='Ya':
        HIV_Infection=1
    elif HIV_Infection=='Tidak':
        HIV_Infection=0

    Sexually_Transmitted_Infection=st.selectbox(
        'Sexually Transmitted Infection',
        ('Ya','Tidak')
    )
    if Sexually_Transmitted_Infection=='Ya':
        Sexually_Transmitted_Infection=1
    elif Sexually_Transmitted_Infection=='Tidak':
        Sexually_Transmitted_Infection=0

    prediksi=st.button("Prediksi")
    if prediksi:
        dataArray = [Systemic_Illness, Rectal_Pain, Sore_Throat, Penile_Oedema, Oral_Lesions, Solitary_Lesion, Swollen_Tonsils, HIV_Infection, Sexually_Transmitted_Infection]
        pred = loaded_model.predict([dataArray])
        if int(pred[0])==0:
            st.success(f"Hasil Prediksi : Negative Terkena Cacar Monyet")
        elif int(pred[0])==1:
            st.error(f"Hasil Prediksi : Positive Terkena Cacar Monyet")
