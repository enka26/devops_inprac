import re
import nltk
from numpy.core.fromnumeric import shape
from pandas.core.frame import DataFrame

import pytest

from os import path
import numpy as np
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords


from predict_class import load_model_
from predict_class import query_lemmatize

from relearn_class import save_model
from relearn_class import load_proc_data
from relearn_class import relearn
from relearn_class import tokenize

PATH_TD = '/home/enka/Downloads/2_cat_data.csv'
PATH_MODEL = './linSVC_tfidf.pkl'
path_to_save_model = './test_save/'
path_to_proc_data = PATH_TD

def load_test_data(path_test_data, size=100):
    assert(path.exists(path_test_data))

    pd_DATA = pd.read_csv(path_test_data)
    pd_DATA = pd_DATA.head(size)
    return (pd_DATA[['name', 'description']], pd_DATA.category_id)
    

class Test1:
    # unittest predict_class 
    def test_load_model(self):
        assert(path.exists(PATH_MODEL))
        class_model = load_model_(PATH_MODEL)
        assert isinstance(class_model, Pipeline)

    
    def test_query_lemmatize_gen_correct(self):
        txt = query_lemmatize('Продам гитару. Гитара практически новая, состояние отличное.')
        assert(txt == 'продавать гитара гитара практически новый состояние отличный')
    
    def test_query_lemmatize_res_txt(self):
        txt = query_lemmatize('Съешь еще этих французских булок да выпей чаю')
        assert(isinstance(txt, str))

    def test_query_lemmatize_send_array(self):
        flag = False
        try:
            txt = query_lemmatize(['Съешь еще этих французских булок да выпей чаю'])
        except AttributeError:
            flag = True

        assert(flag)

    def test_predict_right(self):
        assert(path.exists(PATH_MODEL))
        class_model = load_model_(PATH_MODEL)
        Query = 'Продам гитару. Гитара практически новая, состояние отличное.'
        lem = query_lemmatize(Query)
        assert(int(class_model.predict([lem])) == 50)
    
    def test_predict_wrong(self):
        assert(path.exists(PATH_MODEL))
        class_model = load_model_(PATH_MODEL)
        Query = 'Продам телефон samsung на запчасти.'
        lem = query_lemmatize(Query)
        assert(not int(class_model.predict([lem])) == 50)
    
    
   #unittest relearn_class 


class Test2:
    def test_tokenize_str(self):
        txt = tokenize('Съешь еще этих французских булок да выпей чаю')
        assert(txt == ['Съешь', 'этих', 'французских', 'булок','выпей', 'чаю'] )
    def test_tokenize_nparray(self):
        txt = tokenize(np.array(['Съешь еще этих французских булок да выпей чаю']))
        assert(txt == ['Съешь', 'этих', 'французских', 'булок', 'выпей', 'чаю'] )
    
    def test_tokenize_stopwords(self):
        sw = stopwords.words("russian")
        txt = tokenize(np.array(sw[1:20:2]))
        assert(txt == [])
    
    def test_save_model(self):
        assert(path.exists(PATH_MODEL))
        class_model = load_model_(PATH_MODEL)
        assert(path.exists(path_to_save_model))
        save_model(class_model, 0.7, path_to_save_model)
    
    def test_load_proc_data(self):
        assert(path.exists(path_to_proc_data))
        pdX,y = load_proc_data(path_to_proc_data)
        assert(isinstance(pdX, pd.DataFrame))
        
        assert(shape(pdX)[0] == shape(y)[0])

#все переобучение, может работать долго
class TestRELEARN:
    def test_relearn(self):
        acc = relearn(path_to_proc_data, path_to_save_model)
        assert(acc > 0.6)
    
