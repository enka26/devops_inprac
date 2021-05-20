
from predict_class import *

import pytest
import requests 

# сервер должен работать

class Test:
    def test_send_query(self):
        url= 'http://' + HOST + ':'+PORT+'/predict_class'
        params = {'name': 'Гитара', 'description':'Продам гитару. Гитара практически новая, состояние отличное.'}
        response = requests.get(url, params=params)
        assert(response.json()['prediction'] == 50)
    
    def test_wrong_pred(self):
        url= 'http://' + HOST + ':'+PORT+'/predict_class'
        params = {'name': 'Гитара', 'description':'Продам гитару. Гитара практически новая, состояние отличное.'}
        response = requests.get(url, params=params)
        assert(not response.json()['prediction'] == 0)