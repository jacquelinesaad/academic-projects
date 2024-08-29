"""
    Name: Jacqueline Saad
    Email: jacqueline.saad05@myhunter.cuny.edu
    Resources: Used Pandas and sklearn docs
"""

from typing import Union
import pickle
import pandas as pd
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score

def import_data(file_name):
    '''
    Reads the CSV file into a DataFrame.
    '''
    df = pd.read_csv(file_name, usecols=['VendorID', 'tpep_pickup_datetime',
                                         'tpep_dropoff_datetime', 'passenger_count',
                                         'trip_distance', 'PULocationID', 'DOLocationID',
                                         'fare_amount', 'tip_amount', 'tolls_amount',
                                         'total_amount'])

    # Filter out rows with invalid data.
    df = df[(df['total_amount'] > 0) & (df['trip_distance'] <= 200)]

    return df

def add_tip_time_features(df):
    '''
    Adds tip and time-related features to the DataFrame.
    '''
    # Convert pickup and drop-off times to datetime objects.
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # Compute the duration of the trip in minutes.
    df['duration'] = (df['tpep_dropoff_datetime'] -
                      df['tpep_pickup_datetime']).dt.total_seconds() / 60

    # Compute percent_tip: 100 * tip_amount / (total_amount - tip_amount).
    df['percent_tip'] = 100 * df['tip_amount'] / (df['total_amount'] - df['tip_amount'])

    # Compute the day of the week (0 for Monday, 1 for Tuesday, ... 6 for Sunday).
    df['dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek

    return df

def impute_numeric_cols(df, x_num_cols):
    '''
    Imputes missing values in specified numeric columns with the median.
    '''
    # Replace missing values with the median for each specified numerical column.
    for col in x_num_cols:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    return df

def add_boro(df, file_name):
    '''
    Adds pick-up and drop-off borough information to the DataFrame.
    '''
    # Load the Taxi Zones data from the provided CSV file.
    taxi_zones = pd.read_csv(file_name)

    # Create a mapping of LocationID to borough.
    location_to_borough = dict(zip(taxi_zones.LocationID, taxi_zones.borough))

    # Add PU_borough and DO_borough columns using the mapping.
    df['PU_borough'] = df['PULocationID'].astype(int).map(location_to_borough)
    df['DO_borough'] = df['DOLocationID'].astype(int).map(location_to_borough)

    return df

def add_flags(df):
    '''
    Adds indicators for paid tolls and cross-borough trips to the DataFrame.
    '''
    # Add the paid_toll column based on tolls_amount.
    df['paid_toll'] = (df['tolls_amount'] > 0).astype(int)

    # Add the cross_boro column based on PU_borough and DO_borough.
    df['cross_boro'] = (df['PU_borough'] != df['DO_borough']).astype(int)

    return df

def encode_categorical_col(col, prefix):
    '''
    Encodes a categorical column using one-hot encoding.
    '''
    # Encode the categorical column using get_dummies.
    encoded_df = pd.get_dummies(col, prefix=prefix, prefix_sep='')

    # Drop the last column to keep k-1 columns.
    encoded_df = encoded_df.iloc[:, :-1]

    return encoded_df

def split_test_train(df, xes_col_names, y_col_name, test_size=0.25,
                     random_state=2023) -> Union[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Splits the data into training and testing sets.
    '''
    # Split the data into features (independent variables) and
    # the target variable (dependent variable).
    x_variable = df[xes_col_names]
    y_variable = df[y_col_name]

    # Split the data into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(x_variable, y_variable, test_size=test_size,
                                                        random_state=random_state)

    return x_train, x_test, y_train, y_test

def fit_logistic_regression(x_train, y_train, penalty=None, max_iter=1000, random_state=2023):
    '''
    Fits a logistic regression model to the training data.
    '''
    # Create and train a logistic regression model.
    model = LogisticRegression(penalty=penalty, solver='saga',
                               max_iter=max_iter, random_state=random_state)
    model.fit(x_train, y_train)

    # Serialize the trained model to a bytestream using pickle.
    model_bytestream = pickle.dumps(model)

    return model_bytestream

def fit_svc(x_train, y_train, kernel='rbf', max_iter=1000, random_state=2023):
    '''
    Fits a support vector machine classifier to the training data.
    '''
    # Create and train a support vector machine classifier.
    model = SVC(kernel=kernel, max_iter=max_iter, random_state=random_state)
    model.fit(x_train, y_train)

    # Serialize the trained model to a bytestream using pickle.
    model_bytestream = pickle.dumps(model)

    return model_bytestream

def fit_random_forest(x_train, y_train, num_trees=100, random_state=2023):
    '''
    Fits a random forest classifier to the training data.
    '''
    # Create and train a random forest classifier.
    model = RandomForestClassifier(n_estimators=num_trees, random_state=random_state)
    model.fit(x_train, y_train)

    # Serialize the trained model to a bytestream using pickle.
    model_bytestream = pickle.dumps(model)

    return model_bytestream


def predict_using_trained_model(mod_pkl, xes, yes) -> Union[float, float]:
    '''
    Predicts values using a trained model and calculates mean squared error and r2 score.
    '''
    # Load the trained model from the pickle file.
    with open(mod_pkl, 'rb') as model_file:
        trained_model = pickle.load(model_file)

    # Use the trained model to make predictions.
    predictions = trained_model.predict(xes)

    # Calculate mean squared error and r2 score.
    mse = mean_squared_error(yes, predictions)
    r2_score_value = r2_score(yes, predictions)

    return mse, r2_score_value

def best_fit(mod_list, name_list, xes, yes, verbose=False) -> Union[object, str]:
    '''
    Finds the best-fit model from a list of trained models.
    '''
    best_model = None
    best_model_name = None
    best_r2 = -1.0  # Initialize with a very low value.

    for model, model_name in zip(mod_list, name_list):
        mse, r2_variable = predict_using_trained_model(model, xes, yes)

        if verbose:
            print(f'MSE cost for model {model_name}: {mse:.3f}')

        # Check if the current model has a better r2 score than the best one so far.
        if r2_variable > best_r2:
            best_model = model
            best_model_name = model_name
            best_r2 = r2_variable

    return best_model, best_model_name
