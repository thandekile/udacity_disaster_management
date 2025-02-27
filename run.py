import json
import plotly
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import joblib

app = Flask(__name__)

def tokenize(text):
    """Tokenize and clean text."""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

# Load data and model
db_path = 'sqlite:///../data/DisasterResponse.db'
engine = create_engine(db_path)
df = pd.read_sql_table('disaster_response', engine)
model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    """Render the index page with visualizations."""
    genre_counts = df.groupby('genre')['message'].count()
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    
    graphs = [
        {
            'data': [Bar(x=genre_counts.index, y=genre_counts)],
            'layout': {'title': 'Message Genre Distribution', 'xaxis': {'title': 'Genre'}, 'yaxis': {'title': 'Count'}}
        },
        {
            'data': [Bar(x=category_counts.index, y=category_counts)],
            'layout': {'title': 'Message Category Distribution', 'xaxis': {'title': 'Category'}, 'yaxis': {'title': 'Count'}}
        }
    ]
    
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('master.html', ids=[f"graph-{i}" for i in range(len(graphs))], graphJSON=graphJSON)

@app.route('/go')
def go():
    """Handle user query and display model results."""
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    
    return render_template('go.html', query=query, classification_result=classification_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
