import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Check phishing website", layout="wide")
st.header("""
Welcome to my machine learning dashboard,
""", divider='rainbow')

st.write("created by : [@anggaridwisaputra](https://www.linkedin.com/in/anggari-dwi-saputra-05a51b187/?originalSubdomain=id)")
add_selectitem = st.sidebar.selectbox("Which model do you want to use?", ("Logistic Regression", "Decision Tree", "Random Forest", "Neural Network", "The Best Model"))
st.info("""
    This App predicts whether the website is considered as Phishing or not.
         """)
         
st.write(""" 
    Data to train the model is obtained from:
    - Dataset 1 : Web page phishing detection
    Hannousse, Abdelhakim; Yahiouche, Salima (2021), “Web page phishing detection”, Mendeley Data, V3, doi: 10.17632/c2gw7fy2j4.3

    - Dataset 2: Phishing Websites Dataset
    Vrbančič, Grega (2020), “Phishing Websites Dataset”, Mendeley Data, V1, doi: 10.17632/72ptz43s9v.1
         
    The data are as follows:
    """)

df = pd.read_csv("webpagephishing.csv")
st.dataframe(df, use_container_width=True)

eda1=df.groupby('phishing').agg('mean').reset_index()

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure(figsize=(16,16))
for index,column in enumerate(eda1.iloc[:,1:]):
    plt.subplot(7,3,index+1)
    sns.barplot(data=eda1,x=eda1['phishing'],y=eda1[column], hue=eda1['phishing'])
    plt.xticks(rotation = 90)
plt.tight_layout(pad = 1.0)
st.pyplot()

st.subheader("""Let's Try:sunglasses:""")
url = st.text_input("""input the website address that will be checked. 
                    """)

def predict(url):
    url_length = len(url)
    n_dots = url.count('.')
    n_hypens = url.count('-')
    n_underline = url.count('_')
    n_slash = url.count('/')
    n_questionmark = url.count('?')
    n_equal = url.count('=')
    n_at = url.count('@')
    n_and = url.count('&')
    n_exclamation = url.count('!')
    n_space = url.count(' ')
    n_tilde = url.count('~')
    n_comma = url.count(',')
    n_plus = url.count('+')
    n_asterisk = url.count('*')
    n_hashtag = url.count('#')
    n_dollar = url.count('$')
    n_percent = url.count('%')
    
    features = {
        'url_length': [url_length],
        'n_dots': [n_dots],
        'n_hypens': [n_hypens],
        'n_underline': [n_underline],
        'n_slash': [n_slash],
        'n_questionmark': [n_questionmark],
        'n_equal': [n_equal],
        'n_at': [n_at],
        'n_and': [n_and],
        'n_exclamation': [n_exclamation],
        'n_space': [n_space],
        'n_tilde': [n_tilde],
        'n_comma': [n_comma],
        'n_plus': [n_plus],
        'n_asterisk': [n_asterisk],
        'n_hastag': [n_hashtag],
        'n_dollar': [n_dollar],
        'n_percent': [n_percent]
    }
    
    input_df = pd.DataFrame(features)
    return input_df


if st.button('Predict!'):
        df_predict = predict(url)
        st.info("""
            Below are the summary of your website features
                """)
        st.dataframe(df_predict, use_container_width=True)
        with open("best_phishing_detection.pkl", 'rb') as file:  
             pick = pickle.load(file)
        prediction = pick.predict(df_predict)
        result = ['Be Careful It is detected as Phising' if prediction == 1 else 'It is ok to explore the website']
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(3)
            st.success(f"{output}")