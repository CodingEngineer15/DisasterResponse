import sys
import pickle

import pandas as pd
import numpy as np

from sqlalchemy import create_engine

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def load_data(database_filepath):
    """Loads an sqlite database from the filepath database_filepath"""
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('messages',engine)
    X = df.message
    Y = df.drop(columns=['id','message','original','genre']) 
    
    #Return input, output, category names
    return X,Y, Y.columns


def tokenize(text):
    """Tokenizes a string. To be passed to CountVectorizer"""
	
    stop_words = stopwords.words('english')
	#Regularise the string by making lower case and removing special characters
    text = text.lower()
    text = re.sub("[\W]", ' ',text)
    #Tokenize text
    words = word_tokenize(text)
    #Strip white spaces
    for i in range(len(words)):
        words[i] = words[i].strip()
        
    
    #Lemmatize the text
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words if w not in stop_words]
    lemmed = [WordNetLemmatizer().lemmatize(w,pos='v') for w in lemmed]
    
    return lemmed


def build_model():
    """Builds the pipeline which includes the data transformation the and ML algoithm"""
    params_sgd = {
        'sgd__estimator__alpha': [0.01,0.001,0.0001],
            'sgd__estimator__penalty':['l2', 'l1']
    }
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('sgd',OneVsRestClassifier(SGDClassifier()))
    ])
    
    grid_sgd = GridSearchCV(pipeline,param_grid=params_sgd)

    return grid_sgd


def evaluate_model(model, X_test, Y_test, cat_names):
    """Evaluates the model by returning the f1 score, precision and recall
    for each category under category names

    input
    model:The trained model. Should be a scikit learn pipeline
    X_test: The features of the test set
    Y_test: The labels of the test set
    cat_names: The names of the categories a given message could belong to
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=cat_names))


def save_model(model, model_filepath):
    """Saves the model under a given model_filepath as a pickled file"""
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)



def main():
    """Loads the data used for training and validation, trains a model on the
    training data and validates on the test data and finally saves the 
    resultant model"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
