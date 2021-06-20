import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """This function loads the messages and categories data using the
    respective filepaths as an input (data must be in the csv format)
    and combines them into a merged dataframe
    
    Input
    message_filepath: Filepath to a csv file containing the messages
    categories_filepath: Filepath to a csv file containing the categories
    of the messages e.g flooding

    Output
    merged_df:  A pandas dataframe which has the messages and categories merged
    as into a single data frame
    """

    #Loads messages and categories as dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge dataframe using common id
    merged_df = messages.merge(categories,on='id')
    return merged_df

def clean_data(df):
    """Converts inputs in the category columns into binary values of 0 and 1.
    Then cleans the resultant dataframe of any duplicants and inconsistent
    inputs

    input
    df: A pandas dataframe which contains the message and categories
    
    output:
    df: The cleansed dataframe
    """
    
    #Expand categories column into a dataframe of all the different categories
    categories = pd.DataFrame(df.categories.str.split(';',expand=True))

    #Get the names of each category and rename each category column
    row = categories.iloc[0]
    category_colnames = [string[:-2] for string in row]
    categories.columns = category_colnames

    #Convert the category columns to the numbers 1 and 0.
    for column in categories:
        # The numerical value is the last character of the string
        categories[column] = categories[column].apply(lambda x: int(x[-1]))

    #Drop the redundant categories column and merge the categories dataframe
    #with df
    df.drop(columns=['categories'],inplace=True)
    df = pd.concat([df,categories],axis=1)

    #The related category has 2 as a value. This is likely a flag for
    #ambiguous. Therefore replace with modal value which is 1
    df['related'][df.related == 2] =  1

    #Drop duplicates and return cleansed dataframe
    df.drop_duplicates(inplace=True)
    #print('The number of duplicates are:{}'.format(df.duplicated().sum()))
    #print(df.head(3))
    return df


def save_data(df, database_filename):
    """Sends the dataframe df to an sql database of name database_filename
    input with a table of name messages
    
    input
    df: Pandas dataframe to be converted into a sqlite database
    database_filename: Name of the database to be produced in the same
    directory as this script
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)  


def main():
    """Receives the features under filename messages_filepath and labels
    named under the categories_filepath. Then merges and cleans the data and
    saves the resultant data into a sql filepath"""
 
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
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
