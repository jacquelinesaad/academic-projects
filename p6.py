"""
    Name: Jacqueline Saad
    Email: jacqueline.saad05@myhunter.cuny.edu
    Resources: SQL docs
"""

import pandas as pd
import pandasql as psql

def make_df(file_name):
    '''
    Reads the CSV file into a DataFrame.
    '''

    df = pd.read_csv(file_name)

    # Drop rows with null values in specific columns.
    df = df.dropna(subset=['TYP_DESC', 'INCIDENT_DATE', 'INCIDENT_TIME', 'BORO_NM'])

    return df

def compute_time_delta(start, stop):
    '''
    Converts input strings into datetime objects
    & returns the time difference in seconds.
    '''

    start = pd.to_datetime(start)
    stop = pd.to_datetime(stop)

    time_difference = (stop - start).total_seconds()

    return time_difference

def select_boro_column(_df):
    """
    Selects and returns the BORO_NM column.
    """

    query = 'SELECT "BORO_NM" FROM _df'
    boro_col = psql.sqldf(query)
    return boro_col

def select_by_boro(_df, boro_name):
    """
    Selects and returns all rows where the borough is boro_name.
    """

    boro_name = boro_name.upper()
    query = f'SELECT * FROM _df where BORO_NM == "{boro_name}"'
    filtered_df = psql.sqldf(query)
    return filtered_df

def new_years_count(_df, boro_name):
    """
    Selects and returns the number of incidents called in on 
    New Year's Day (Jan 1, 2021) in the specified borough.
    """

    boro_name = boro_name.upper()
    query = f'''SELECT COUNT(*) FROM (SELECT * FROM _df WHERE INCIDENT_DATE == "01/01/2021"
    AND BORO_NM == "{boro_name}") AS subquery'''
    filtered_df = psql.sqldf(query)
    return filtered_df

def incident_counts(_df):
    """
    Selects and returns the incident counts per radio code (TYP_DESC),
    sorted alphabetically.
    """

    query = 'SELECT TYP_DESC, COUNT(*) FROM _df GROUP BY TYP_DESC ORDER BY TYP_DESC'
    filtered_df = psql.sqldf(query)
    return filtered_df

def top_10(_df, boro_name):
    """
    Selects and returns the top 10  incidents by radio code
    in the specified borough.
    """

    boro_name = boro_name.upper()
    query = f'''SELECT TYP_DESC, COUNT(*) FROM (SELECT * FROM _df WHERE BORO_NM == "{boro_name}")
    GROUP BY TYP_DESC ORDER BY COUNT(*) DESC LIMIT 10'''
    filtered_df = psql.sqldf(query)
    return filtered_df
