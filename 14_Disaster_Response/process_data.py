# Import Libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Load Messages
messages = pd.read_csv('data/disaster_messages.csv')
messages.head()

# Load Categories
categories = pd.read_csv('data/disaster_categories.csv')
categories.head()

print(categories['categories'][0])

# 2. Merge datasets.
# Merge the messages and categories datasets using the common id
# Assign this combined dataset to df, which will be cleaned in the following steps

# Merge Datasets
df = messages.merge(categories, on=['id'])
print(df.head())

# 3. Split categories into separate category columns.
# Split the values in the categories column on the ; character so that each value becomes a separate column. You'll find this method very helpful! Make sure to set expand=True.
# Use the first row of categories dataframe to create column names for the categories data.
# Rename columns of categories with new column names.

# Create a DataFarme of the 36 Individual Category Columns
categories = df['categories'].str.split(';', expand=True)
print(categories.head())

# Select the First Row of the Categories DataFrame
row = categories.head(1)
categories_colnames = row.applymap(lambda x: x[:-2]).iloc[0,:].tolist()
print(categories_colnames)

# Rename the Columns of 'Categories'
categories.columns = categories_colnames
categories.head()

# 4. Convert category values to just numbers 0 or 1.¶
# Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, related-0 becomes 0, related-1 becomes 1. Convert the string to a numeric value.
# You can perform normal string actions on Pandas Series, like indexing, by including .str after the Series. You may need to first convert the Series to be of type string, which you can do with astype(str).

for column in categories:
    # Set Each Value to be the Last Character of the String
    categories[column] = categories[column].astype(str).str[-1]
    
    # Convert Column from String to Numeric
    categories[column] = categories[column].astype(int)
    
print(categories.head())

# 5. Replace categories column in df with new category columns.
# Drop the categories column from the df dataframe since it is no longer needed.
# Concatenate df and categories data frames.

# Drop the Original Categories Column from df
df.drop('categories', axis=1, inplace=True)
print(df.head())

# Concatenate the Original DataFrame with the new 'Categories' DataFrame
df = pd.concat([df,categories], axis=1)
print(df.head())

#### 6. Remove duplicates.
# * Check how many duplicates are in this dataset.
# * Drop the duplicates.
# * Confirm duplicates were removed.

print(f'Count of Duplicates: {df[df.duplicated()].count()}')

# Check Number of Duplicates
print(df[df.duplicated()].shape)

# Drop Duplicates
print(df.drop_duplicates(inplace=True))

# Check Number of Duplicates 
print(df[df.duplicated()].count())

engine  = create_engine('sqlite:///data/disaster_db.db')
df.to_sql('messages_disaster', engine, index=False)