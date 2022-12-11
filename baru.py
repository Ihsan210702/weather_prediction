import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
import hydralit_components as hc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import altair as alt


with st.sidebar:
# specify the primary menu definition
    menu_data = [
            {'icon': "far fa-clipboard", 'label':"Description"},
            {'icon':"far fa-chart-bar",'label':"Datasets"},
            {'icon':"fas fa-balance-scale",'label':"Prepocessing"},
            {'icon': "fas fa-save", 'label':"Modelling"},#no tooltip message
            {'icon': "fas fa-edit", 'label':"Implementation"},
    ]
    # we can override any part of the primary colors of the menu
    #over_theme = {'txc_inactive': '#FFFFFF','menu_background':'red','txc_active':'yellow','option_active':'blue'}
    over_theme = {'txc_inactive': '#FFFFFF','menu_background':'#89CFF0'}
    menu_id = hc.nav_bar(menu_definition=menu_data,home_name='Home',override_theme=over_theme,hide_streamlit_markers=False, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned',)
    st.markdown("<h1 style='font-size:15px;text-align: center; color: gray;'>Fajrul Ihsan Kamil_200411100172</h1>", unsafe_allow_html=True)
if menu_id == "Home":
    st.markdown("""<h1 style='text-align: center;'> Weather Prediction Application </h1> """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")

    with col2:
        img = Image.open('cuaca.jpg')
        st.image(img, use_column_width=False, width=300)

    with col3:
        st.write("")

    st.write(""" """)
    
    
    st.write(""" """)

    st.write("""
    Weather Forecasting, dalam bahasa sehari-hari disebut ramalan cuaca, adalah penggunaan ilmu dan teknologi untuk memperkirakan 
    keadaan atmosfer Bumi pada masa datang untuk suatu tempat tertentu. 
    """)
    st.write("""
    Saat ini prakiraan cuaca dilakukan menggunakan pemodelan (modeling) dengan bantuan komputer. Walaupun sudah dibantu teknologi, 
    tetapi keakuratan tidak dapat mencapai 100% dan masih ada kemungkinan salah.
    """)
if menu_id == "Description":
    st.subheader("Pengertian")
    st.write(""" Dataset ini merupakan data Weather Prediction (Prediksi Cuaca) yang diambil dari website resmi kaggle.com. Data Prediksi Cuaca ini
    menampilkan data dari tahun 2012 sampai 2015. Selanjutnya data ini nantinya akan digunakan untuk melakukan prediksi Cuaca di hari-hari berikutnya. Dataset ini
    sendiri terdiri dari 6 atribut yaitu Date, Precipitation, temp_max, temp_min, wind, dan weather """)
    st.subheader("Kegunaan Dataset")
    st.write(""" Dataset ini digunakan untuk melakukan prediksi cuaca di hari hari berikutnya. Setelah dilakukan implementasi selanjutnya akan langsung dapat memprediksi
    cuaca dengan hasil weather yang sudah ditentukan. """)
    st.subheader("Fitur")
    st.markdown(
            """
            Fitur-fitur yang terdapat pada dataset ini :
            - Date : Berisi tentang tanggal berapa cuaca tersebut diprediksi
            - Precipitation 
                - Precipitation atau Curah Hujan adalah kondensasi uap air atmosfer yang jatuh di bawah tarikan gravitasi dari awan.
            - Temp_max 
                - Temperature maximum atau Suhu maksimum suatu daerah adalah suhu maksimum dari daerah itu dalam jangka waktu tertentu. Suhu maksimum yang pernah tercatat di Bumi adalah 56 derajat Celcius pada daerah California. Suhu maksimum pada hari tersebut cenderung terjadi pada siang hari atau sore hari.
            - Temp_min  
                - Temperature minimum atau Suhu minimum suatu daerah adalah suhu terendah dari daerah itu dalam jangka waktu tertentu. Suhu serendah mungkin adalah -273 derajat Celcius, atau 0 Kelvin (Kelvin adalah satuan suhu). Suhu minimum biasa terjadi pada dini hari.
            - Wind 
                - Wind atau Kecepatan angin adalah pergerakan udara, yang disebabkan oleh pemanasan bumi yang tidak merata oleh matahari dan rotasi bumi itu sendiri. 
            - Weather 
                - Berisi tentang hasil prediksi cuaca yang didalamnya terdapat beberapa cuaca yaitu :
                    - Drizzle
                    - Fog
                    - Rain
                    - Snow
                    - Sun
            """
        )
    st.subheader("Sumber Dataset")
    st.write("""
        Sumber data di dapatkan melalui website kaggle.com, Berikut merupakan link untuk mengakses sumber dataset.
        <a href="https://www.kaggle.com/datasets/ananthr1/weather-prediction">Klik disini</a>""", unsafe_allow_html=True)
        
    st.subheader("Tipe data")
    st.write("""
        Tipe data yang di gunakan pada dataset anemia ini adalah NUMERICAL.
        """)
if menu_id == "Datasets":
    st.markdown("""<h2 style='text-align: center; color:grey;'> Dataset Weather Prediction </h1> """, unsafe_allow_html=True)
    df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/seattle-weather.csv')
    c1, c2, c3 = st.columns([1,5,1])

    with c1:
        st.write("")

    with c2:
        df

    with c3:
        st.write("")
if menu_id == "Prepocessing":
    st.header("""Normalisasi Data""")
    st.write("""Rumus Normalisasi Data :""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Dimana :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    st.subheader("""Data Awal""")
    df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/seattle-weather.csv')
    df

    df = df.drop(columns=['date'])
    #Membuat variable x dimana variable label di hilangkan
    X = df.drop(columns=['weather'])
    a = df.weather

    st.subheader("""Hasil Normalisasi Data""")
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)
    scaled_features

    #Encoder
    st.subheader("""Hasil Encoder Label""")
    le = preprocessing.LabelEncoder()
    le.fit(a)
    s = le.transform(a)
    labels = pd.get_dummies(df.weather).columns.values.tolist()  
    labels
    y = pd.DataFrame(s, columns =['weather'])
    result = pd.concat([scaled_features, y], axis=1)
    result
if menu_id == "Modelling":
    #Read Dataset
    df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/seattle-weather.csv')

    #Preprocessing data
    #Mendefinisikan Varible X dan Y

    df= df.drop(columns=['date'])
    X = df.drop(columns=['weather'])
    a = df.weather
    le = preprocessing.LabelEncoder()
    le.fit(a)
    s = le.transform(a)
    y = pd.DataFrame(s, columns =['weather'])
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    #Split Data 
    training, test = train_test_split(scaled_features,test_size=0.3, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.3, random_state=1)#Nilai Y training dan Nilai Y testing
    
    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighboor')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        #Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        probas = gaussian.predict(test)
   
        gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier(random_state=1)
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("Model KNN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Model Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik akurasi semua model")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)
if menu_id=="Implementation":
    #Read Dataset
    df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/dataset/main/seattle-weather.csv')

    #Preprocessing data
    #Mendefinisikan Varible X dan Y

    df= df.drop(columns=['date'])
    X = df.drop(columns=['weather'])
    a = df.weather
    le = preprocessing.LabelEncoder()
    le.fit(a)
    s = le.transform(a)
    y = pd.DataFrame(s, columns =['weather'])
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    #Split Data 
    training, test = train_test_split(scaled_features,test_size=0.3, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.3, random_state=1)#Nilai Y training dan Nilai Y testing
    
    #Gaussian Naive Bayes
    gaussian = GaussianNB()
    gaussian = gaussian.fit(training, training_label)
    probas = gaussian.predict(test)

    #KNN
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(training,training_label)
    knn_predict=knn.predict(test)

    #Decission Tree
    dt = DecisionTreeClassifier(random_state=1)
    dt.fit(training, training_label)
    # prediction
    dt_pred = dt.predict(test)
    with st.form("my_form"):
        st.header("Implementation")
        Precipitation = st.number_input('Masukkan preciptation (curah hujan) : ')
        Temp_Max = st.number_input('Masukkan tempmax (suhu maks) : ')
        Temp_Min = st.number_input('Masukkan tempmin (suhu min) : ')
        Wind = st.number_input('Masukkan wind (angin) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                    ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        prediksi = st.form_submit_button("Submit")
        if prediksi:
            inputs = np.array([
                Precipitation,
                Temp_Max,
                Temp_Min,
                Wind
            ])

            df_min = X.min()
            df_max = X.max()
            input_norm = ((inputs - df_min) / (df_max - df_min))
            input_norm = np.array(input_norm).reshape(1, -1)

            if model == 'Gaussian Naive Bayes':
                mod = gaussian
                akurasi = round(100 * accuracy_score(test_label,probas))
            if model == 'K-NN':
                mod = knn 
                akurasi = round(100 * accuracy_score(test_label,knn_predict))
            if model == 'Decision Tree':
                mod = dt
                akurasi = round(100 * accuracy_score(test_label,dt_pred))

            input_pred = mod.predict(input_norm)

            st.subheader('Hasil Prediksi')
            st.write('Menggunakan Pemodelan :',model)
            st.write('Akurasi: {0:0.2f}'. format(akurasi))
            

            if input_pred == 4:
                st.success('Sun')
            elif input_pred == 3:
                st.success('Snow')
            elif input_pred == 2:
                st.success('Rain')
            elif input_pred == 1:
                st.success('Fog')
            else:
                st.success('Drizzle')