import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../Classifier.pkl")



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #Calculate the top categories and counters
    top_cat_count = df.iloc[:,4:].mean().sort_values(ascending=True)[1:30]
    top_cat_names = list(top_cat_count.index)
    aid_counts = df.groupby('aid_related').count()['message']
    aid_names = list(aid_counts.index)
    cat_related_names = ['aid_related', 'weather_related', 'infrastructure_related']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=cat_related_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Aid Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "AID Related"
                }
            }
        },

       {
            'data': [
                Bar(
                    x=top_cat_names,
                    y=top_cat_count
                )
            ],

            'layout': {
                'title': 'Bar Plot of the Disaster Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Disaster Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)

if __name__ == '__main__':
    main()