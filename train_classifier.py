import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
import pickle

def load_data(database_filepath):
    """
    Load data from the SQLite database and split it into features (X) and target (Y).

    Parameters:
    database_filepath (str): Path to the SQLite database.

    Returns:
    X (pd.DataFrame): Feature data (messages).
    Y (pd.DataFrame): Target data (categories).
    category_names (list): List of category names.
    """
    # Create a connection to the database
    engine = create_engine(f'sqlite:///{database_filepath}')
    
    # Load the data into a dataframe from the 'disaster_response' table
    df = pd.read_sql('SELECT * FROM disaster_response', engine)

    # Split the data into X (messages) and Y (categories)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])

    # Get category names
    category_names = Y.columns.tolist()

    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and preprocess the text data.

    Parameters:
    text (str): The input text message.

    Returns:
    tokens (list): The list of tokens (words).
    """
    # Tokenize text using TfidfVectorizer (which includes lowercasing, punctuation removal, etc.)
    # Additional preprocessing steps can be added here
    return text.split()

def build_model():
    """
    Build and return a machine learning model using a pipeline.

    Returns:
    model (sklearn.pipeline.Pipeline): A machine learning pipeline with Tfidf and RandomForestClassifier.
    """
    # Define the pipeline with TF-IDF vectorizer and RandomForestClassifier wrapped in MultiOutputClassifier
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model's performance on the test set.

    Parameters:
    model (sklearn.pipeline.Pipeline): The trained model.
    X_test (pd.DataFrame): The test feature data (messages).
    Y_test (pd.DataFrame): The test target data (categories).
    category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    
    # Evaluate the model for each category
    for i, category in enumerate(category_names):
        print(f'Category: {category}')
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the trained model to a file.

    Parameters:
    model (sklearn.pipeline.Pipeline): The trained model.
    model_filepath (str): Path to save the trained model.
    """
    # Save the model to a pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # Split the data into train and test sets (80% training, 20% testing)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
