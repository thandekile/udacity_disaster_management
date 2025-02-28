import sys
import pandas as pd
from sqlalchemy import create_engine
 
 
def drop_duplicates_and_filter(df):
    """
    Remove duplicate entries and filter out rows where 'related' column has a value of 2.
    Args:
    df (pd.DataFrame): Input DataFrame.
    Returns:
    pd.DataFrame: Cleaned DataFrame.
    """
    return df.drop_duplicates().query("related != 2")
 
 
def extract_categories(category_series):
    """
    Extract and convert category columns from a single concatenated string column.
    Args:
    category_series (pd.Series): Series containing category strings.
    Returns:
    pd.DataFrame: A DataFrame with categories as separate columns.
    """
    category_dict = {}
    for entry in category_series:
        for pair in entry.split(';'):
            key, val = pair.split('-')
            category_dict.setdefault(key, []).append(int(val))
    return pd.DataFrame(category_dict)
 
 
def load_and_merge_data(messages_path, categories_path):
    """
    Load messages and categories, then merge them into a single DataFrame.
    Args:
    messages_path (str): Filepath for messages dataset.
    categories_path (str): Filepath for categories dataset.
    Returns:
    pd.DataFrame: Merged DataFrame with processed category columns.
    """
    messages_df = pd.read_csv(messages_path)
    categories_df = pd.read_csv(categories_path)
    merged_df = pd.merge(categories_df, messages_df, on='id')
    # Process categories and merge
    expanded_categories = extract_categories(merged_df['categories'])
    merged_df.drop(columns=['categories'], inplace=True)
    return pd.concat([merged_df, expanded_categories], axis=1)
 
 
def save_to_database(df, db_filename):
    """
    Save DataFrame to a SQLite database.
    Args:
    df (pd.DataFrame): Processed DataFrame to be saved.
    db_filename (str): Database filename.
    """
    engine = create_engine(f'sqlite:///{db_filename}')
    df.to_sql('ProcessedMessages', engine, index=False, if_exists='replace')
 
 
def main():
    if len(sys.argv) == 4:
        msg_filepath, cat_filepath, db_filepath = sys.argv[1:]
        print(f'Loading data...\n    MESSAGES: {msg_filepath}\n    CATEGORIES: {cat_filepath}')
        df = load_and_merge_data(msg_filepath, cat_filepath)
        print('Cleaning data...')
        df = drop_duplicates_and_filter(df)
        print(f'Saving data...\n    DATABASE: {db_filepath}')
        save_to_database(df, db_filepath)
        print('Data successfully saved to the database!')
    else:
        print("Usage: python process_data.py <messages.csv> <categories.csv> <database.db>")
 
 
if __name__ == '__main__':
    main()
