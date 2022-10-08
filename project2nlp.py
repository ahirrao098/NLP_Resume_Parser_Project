
import pandas as pd
import numpy as np
import docx2txt,textract
import streamlit as st
import pdfplumber
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot  as plt
stop=set(stopwords.words('english'))
from pickle import load
import pickle
model=load(open(r'C:/Users/pune/Desktop/Streamlit2NLP/New folder/Resume_Classification_NLP-main/New folder/onevsrest.sav','rb'))
vectors = pickle.load(open(r'C:/Users/pune/Desktop/Streamlit2NLP/New folder/Resume_Classification_NLP-main/New folder/tfidf.pkl','rb'))

nltk.download('wordnet')
nltk.download('stopwords')



resume = []

def display(doc_file):
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else :
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
            
    return resume
    



def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words)  




def main():
    menu = ["Prediction","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    html_temp = """
    <div style ="background-color:green;padding:5px">
    <h1 style ="color:white;text-align:center;"> Resume Classification </h1>
    </div>
    <style>
    body {background-image: url("https://unsplash.com/photos/XZdgKvOqsMI");
    background-size: cover;}
    </style>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    
    if choice == "Prediction page":
        st.subheader("Prediction Application")
    upload_file = st.file_uploader('Please Upload a Resume you want to predict for ',
                                type= ['docx','pdf'],accept_multiple_files=True)
        
    if st.button("Predict"):
        for doc_file in upload_file:
            if doc_file is not None:
               
                file_details = {'filename':[doc_file.name],
                               'filetype':doc_file.type.split('.')[-1].upper(),
                               'filesize':str(doc_file.size)+' KB'}
                file_type=pd.DataFrame(file_details)
                st.write(file_type.set_index('filename'))
                displayed=display(doc_file)
                cleaned=preprocess(display(doc_file))
                predicted= model.predict(vectors.transform([cleaned]))

                if int(predicted) == 0:
                    st.header("The Resume Is From Peoplesoft Resumes")
                elif int(predicted) == 1:
                    st.header("The Resume Is From React Js Developers ")
                elif int(predicted) == 2:
                    st.header("The Resume Is From  SQL Developer")
                else:
                    st.header("The Resume Is From  Workday Resumes ")


    elif choice == "About":
        st.header("About") 
        st.subheader("This is a Resume Classification by group 6")
        st.info("Rohit Ahirrao")
        st.info("Sushmita Singh")
        st.info("Kiran Koti")
        st.info("Savita Lodiwale")
        st.info("Akansha Mishra")
          
    
if __name__ == '__main__':
     main()