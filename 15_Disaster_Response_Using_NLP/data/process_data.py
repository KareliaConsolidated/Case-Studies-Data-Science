# Import Necessary Libraries
import os
import pandas as pd
import numpy as np
import sys
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
	
	"""load_data function loads the messages and categories dataframe and return the merged dataframe.
	
	Input:
	messages_filepath : Messages Dataset File Path
	categories_filepath : Categories Dataset File Path

	Output:
	df: Merged dataset from messages and categories
	"""
	# Load Messages Dataset
	messages = pd.read_csv(messages_filepath)

	# Load Categories Dataset
	categories = pd.read_csv(categories_filepath)


	# Merge Datasets
	df = pd.merge(messages, categories, on='id')

	return df

def clean_data(df):
	"""clean_data : It cleans the dataframe categories, by performing ETL step, starts by extracting dataframe of the 36 individual categories, converting and then concatenating followed by removing duplicates
	Input:
	df: Merged Dataset from messages and categories
	Output:
	df: A complete cleaned dataset
	"""

	# Creating a DataFrame of the 36 Individual Category Columns
	categories = df.categories.str.split(';', expand=True)

	# Extracting Column Names 
	row = categories.iloc[0,:]
	categories_colnames = row.apply(lambda x: x[:-2])
	categories.columns = categories_colnames

	# Converting Categories Values to Numeric
	for column in categories:
		categories[column] = categories[column].apply(lambda x: x[-1]).astype(int)

	# Drop categories column in df
	df = df.drop('categories', axis=1)

	# Concatenate the dataframe(df) with the new `categories` dataframe
	df = pd.concat([df, categories], axis=1)

	# Drop Duplicates in df DataFrame
	df.drop_duplicates(inplace=True)

	return df

def save_data(df, database_filename):
	"""Saves the Cleaned Data in DB
	Input : df -> Cleaned DataFrame, database_filename -> Data Will be stored in this database
	Output : A SQLite database
	"""

	# Save df into SQLite Database
	engine = create_engine('sqlite:///'+database_filename)
	df.to_sql('Messages', engine, index=False, if_exists='replace')	

def main():
    """Main function which performs ETL on the datasets"""
    if len(sys.argv) == 4:
    	messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

    	print(f'Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}')
    	df = load_data(messages_filepath, categories_filepath)

    	print('Cleaning data...')
    	df = clean_data(df)

    	print(f'Saving data...\n    DATABASE: {database_filepath}')
    	save_data(df, database_filepath)

    	print('Cleaned data saved to database!')
    else:
    	print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
	main()	