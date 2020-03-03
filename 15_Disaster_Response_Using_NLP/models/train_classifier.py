# Import Libraries
import sys
import pickle
import sqlite3
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'wordnet'])
# import warnings
# warnings.simplefilter('ignore')
import subprocess
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

# Function to load data from database as dataframe

def load_data(database_filepath):
    '''
    Input: database_filepath: File path of sql database
    Output:
        X: Message data (features)
        y: Categories (target)
        col_names: Labels for 36 categories
    '''
    # Load data from database as dataframe
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    y = df.iloc[:, 4:]

    # Listing the Columns
    category_names = list(df.columns[4:])

    return X, y, category_names

#  Function to Tokenize and Clean Text


def tokenize(text):
    '''
    Input: text: Original disaster message text from dataframe
    Output: lemmed: list of text that has been Tokenized, Cleaned, and Lemmatized
    '''

    # Lower case normalization and remove punctuation characters

    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # Splits text into list of words

    words = word_tokenize(text)

    # Remove Stop Words

    words = [w for w in words if w not in stopwords.words('english')]

    # Initiate Lemmatizer

    lemmatizer = WordNetLemmatizer()

    # Chain Lemmatization of Nouns then Verbs

    clean_tokens = [lemmatizer.lemmatize(w, pos='n') for w in words]
    clean_tokens = [lemmatizer.lemmatize(w, pos='v') for w in clean_tokens]

    return clean_tokens         
            

# Function to build the ML Pipeline using CountVectorizer, tfidf, Random Forest, and GridSearch
def build_model():
    '''
    Input: None
    Output: cv_grid of the results of GridSearchCV
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])


    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']}

    cv = GridSearchCV(pipeline, verbose=2, param_grid=parameters, cv=2, n_jobs=1)

    return cv

# Function to Evaluate Model Performance using Test Data
def evaluate_model(model, X_test, y_test, category_names):
    '''
    Input: 
        model: The model to be evaluated
        X_test: The test data of the features
        y_test: The true labels for Test data from split dataset
        category_names: The labels for all 36 categories
    Output:
        Print of accuracy score and classfication report for each category
    '''
    y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(f'Accuracy of {category_names[i]:<25}: {round(accuracy_score(y_test.iloc[:, i].values, y_pred[:,i]),3):>2}%')

    for i in range(len(category_names)):
        print(f"Category: {category_names[i]:<25} \n {classification_report(y_test.iloc[:, i].values, y_pred[:, i])}")

# Function to save model as a pickle file         
def save_model(model, model_filepath):
    '''
    Input: 
        model: The model to save
        model_filepath: path of the output pick file
    Output: A pickle file of saved model
    '''
    pickle.dump(model, open(model_filepath, "wb"))        

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()    