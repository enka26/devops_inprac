#-------------------Configs-------------------
clf_path = './linSVC_tfidf.pkl' 
category_names = './cat2.csv'
HOST = 'localhost'
PORT='8383'
#-------------------Imports-------------------
from flask import Flask
from flask_restful import reqparse, Api, Resource
import pandas as pd
import joblib


import cl_preprocessing as prep
#----------------End of Imports---------------

def load_model_(clf_path):
    with open(clf_path, 'rb') as f:
        class_model = joblib.load(f)
    return class_model

app = Flask(__name__)
api = Api(app)


# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('name')
parser.add_argument('description')


def query_lemmatize(text):
    res = prep.lemmatize(text)
    return res


class PredictClass(Resource):
    def get(self):
        
        class_model = load_model_(clf_path)
        args = parser.parse_args()
        user_query = args['name'] + ' ' + args['description'] 
        
        # нормализация запроса
        user_query = query_lemmatize(user_query)
        prediction = class_model.predict([user_query]) #text over rows format instead of just string
        
        
        category_id = int(prediction[0])
        
        # # по номеру - тестом 
        #categories = pd.read_csv(category_names)
        #category_name = categories.loc[categories['category_id'] ==category_id].category_name.values[0]
        
        output = {'prediction': category_id} #category_name - для текста
        
        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictClass, '/predict_class')


if __name__ == '__main__':
    app.run(host=HOST, port=PORT)
