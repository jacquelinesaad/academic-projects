"""
    Name: Jacqueline Saad
    Email: jacqueline.saad05@myhunter.cuny.edu
    Resources: Inferential Thinking, ThinkCS
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV, LassoCV



def import_data(file_name, names=None):
    '''
    Reads the CSV file into a DataFrame.
    '''
    df = pd.read_csv(file_name)

    # Drop the 'DATE' column
    df = df.drop(columns=['DATE'])

    # If names dictionary is provided, rename columns.
    if names is not None:
        df = df.rename(columns=names)

    # Convert non-numeric entries to NaN and drop the NaN rows.
    df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    df = df.dropna()

    return df

def split_data(df, xes_col_names, y_col_name, test_size=0.25, random_state=1870):
    """
    Splits the data into training and testing sets.
    Returns the results.
    """
    xes = df[xes_col_names]
    yes = df[y_col_name]

    # Split the data into training and testing sets.
    x_train, x_test, y_train, y_test = train_test_split(
        xes, yes, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test

def fit_lin_reg(x_train, y_train):
    """
    Fits a multiple linear regression model to the training data.
    Returns the model as a bytestream using pickle.
    """
    # Create and fit the linear regression model.
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Use pickle to return model as a bytestream.
    model_bytestream = pickle.dumps(model)

    return model_bytestream

def encode_poly(df, x_col, deg=2):
    """
    Generates polynomial features of a column in a DataFrame.
    Returns the resulting array.
    """
    # Extract values as numpy array.
    x_values = df[x_col].to_numpy().reshape(-1, 1)

    # Transform data to generate polynomial features.
    poly = PolynomialFeatures(degree=deg)
    x_poly = poly.fit_transform(x_values)

    return x_poly

def fit_poly(xes, yes, epsilon=0.01, verbose=False):
    """
    Fits polynomial models to data and returns the best model's degree and coefficients.
    """
    best_degree = None
    best_model = None

    for deg in range(1, 6):
        # Generate polynomial features.
        #x_poly = encode_poly(xes, 'units', deg)
        poly_features = PolynomialFeatures(degree=deg, include_bias=False)
        x_poly = poly_features.fit_transform(xes)

        # Create linear regression model. Fit model to data.
        model = LinearRegression(fit_intercept=False)
        model.fit(x_poly, yes)

        # Then make predictions using the model and calculate MSE.
        predictions = model.predict(x_poly)
        error = mean_squared_error(yes, predictions)

        if verbose:
            print(f'MSE cost for deg {deg} poly model: {error:.3f}')

        if error < epsilon:
            best_degree = deg
            best_model = model.coef_
            break

    if best_degree is None:
        return None

    return best_degree, best_model

def fit_with_regularization(xes, yes, poly_deg=2, reg="lasso"):
    """
    Fits a model with polynomial features using Lasso or Ridge regression with cross-validation.
    Returns the model as a serialized object (pickled object).
    """
    # Generate polynomial features.
    x_poly = encode_poly(xes, 'units', poly_deg)

    # RidgeCV Model.
    if reg == "ridge":
        model = RidgeCV(alphas=np.logspace(-6, 6, 13))
    # LassoCV Model.
    elif reg == "lasso":
        model = LassoCV(alphas=np.logspace(-6, 6, 13))
    else:
        raise ValueError("Invalid type. Use 'ridge' or 'lasso'.")

    # Fit model to data.
    model.fit(x_poly, yes)

    # Serialize and return the model.
    model_serialized = pickle.dumps(model)

    return model_serialized

def predict_using_trained_model(mod_pkl, xes, yes):
    """
    Predicts target values using a trained model
    and calculates mean squared error (MSE) and R-squared (R2) score.
    """

    # Load the trained model from pickle.
    loaded_model = pickle.loads(mod_pkl)

    # Make predictions using the loaded model.
    predictions = loaded_model.predict(xes)

    # Calculate mean squared error.
    mse = mean_squared_error(yes, predictions)

    # Calculate R-squared score.
    r_squared = r2_score(yes, predictions)

    # Return MSE and R-squared score as a tuple.
    return mse, r_squared
