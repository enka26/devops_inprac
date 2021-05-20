#-------------------Configs-------------------
HOST = 'localhost'
# TODO: переписать
USER = '' 
PWD = '' 
DATABASE = '' 
things_table_name = '' 
PATH = '' # path to load model
PATH_TO_PREP_DATA = '' # path to prepared data

#-------------------Imports-------------------
from os import path
import numpy as np 
import pandas as pd

from nltk.corpus import stopwords
from string import punctuation

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import joblib
#----------------End of Imports---------------
stop_words = stopwords.words("russian")

def tokenize(text):
    
    if isinstance(text, np.ndarray):
      # np array
      str_text = ''
      for s in text:
        str_text = str_text + ' '+s
      text = str_text
    

    tokens = [token for token in text.split() if token not in stop_words and token != " " \
                      and token.strip() not in punctuation]
    return tokens


def save_model(pipeline, accuracy, path_to_save):
    with open(path_to_save + 'linSVC_tfidf.pkl','wb') as f:
      joblib.dump(pipeline,f)
    
    with open(path_to_save+ 'acc.pkl', 'wb') as a:
      joblib.dump(accuracy, a)

def load_proc_data(path_to_proc):
    pd_DATA = pd.read_csv(path_to_proc)
    return (pd_DATA[['name', 'description']], pd_DATA.category_id)
# ------------------Main function--------------------
def relearn(path_to_data, path_to_save):
    proc_data, target = load_proc_data(path_to_data)

    #learn
    X_train, X_valid, y_train, y_valid = train_test_split(
      proc_data.values, target, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(tokenizer=tokenize, lowercase=False)
    model = LinearSVC()
    pipeline = make_pipeline(vectorizer, model)
    pipeline.fit(X_train, y_train)
	# accuracy
    prediction = pipeline.predict(X_valid)
    accuracy = accuracy_score(prediction, y_valid)

    print(accuracy)
    save_model(pipeline, accuracy, path_to_save)
    return accuracy




