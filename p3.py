"""
Katherine St. John
katherine.stjohn@hunter.cuny.edu
Program 3, Fall 2023
Resources:  Used Pandas & sklearn docs for syntax
"""

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, r2_score



def import_data(file_name,names=None):
    """
    Reads in a file, cleans up columns, and returns a DataFrame.

    The DATE column is dropped and the remaining columns renamed
    with using the parameter names.

    The resulting DataFrame is returned.
    """

    df = pd.read_csv(file_name)
    df = df.drop(columns=['DATE'])
    df = df.apply(lambda col: pd.to_numeric(col, errors ='coerce'))
    df = df.dropna()
    if names is not None:
        df = df.rename(columns = names)

    return df

def split_data(df, xes_col_names, y_col_name, test_size = 0.33, random_state = 106):
    """
    Splits the df into testing and training sets.
    Returns the results.
    """
    x_train, x_test, y_train, y_test = train_test_split(df[xes_col_names],
                                                        df[y_col_name],
                                                        test_size=test_size,
                                                        random_state=random_state)
    return x_train, x_test, y_train, y_test


def fit_lin_reg(x_train, y_train):
    """
    :param x_train: an array of numeric columns with no null values.
    :param y_train: an array of numeric columns with no null values.
    :return: pickled model object
    """
    mod = LinearRegression()
    mod.fit(x_train, y_train)
    pmod = pickle.dumps(mod)
    return pmod

def predict_using_trained_model(mod_pkl, poly_xes, yes):
    """
    Predict and compare model outcomes for x with actual y
    :param mod: a trained model for the data.
    :param x:  an array or DataFrame of numeric columns with no null values.
    :param y:  an array or DataFrame of numeric columns with no null values.
    :return: the mean squared error and r2 score
    """
    y_true = yes

    mod = pickle.loads(mod_pkl)
    y_pred = mod.predict(poly_xes)
    mse = mean_squared_error(y_true, y_pred)
    r_2 = r2_score(y_true, y_pred)
    return mse, r_2

def encode_poly(df, x_col,deg=2):
    """
    ADD IN DOCSTRING
    """
    transformer = PolynomialFeatures(degree=deg)
    x_poly = transformer.fit_transform(df[[x_col]].to_numpy())
    return x_poly

def fit_poly(xes,yes,epsilon=0.01, verbose=False):
    """
    ADD IN DOCSTRING
    """
    error = 2*epsilon
    deg = 0
    while (error > epsilon) and deg < 5:
        deg = deg+1
        transformer = PolynomialFeatures(degree=deg)
        x_poly = transformer.fit_transform(xes)
        clf = LinearRegression(fit_intercept=False)
        clf.fit(x_poly, yes)
        pred_poly = clf.predict(x_poly)
        error = mean_squared_error(pred_poly, yes)
        if verbose:
            print(f'MSE cost for deg {deg} poly model: {error:.3f}')
    if deg == 5:
        return None
    return deg

def fit_with_regularization(xes, yes, poly_deg=2, reg="lasso"):
    """
    ADD IN DOCSTRING
    """
    transformer = PolynomialFeatures(degree= poly_deg)
    x_poly = transformer.fit_transform(xes)
    if reg == "ridge":
        mod = RidgeCV().fit(x_poly,yes)
    else:
        mod = LassoCV().fit(x_poly,yes)
    return pickle.dumps(mod)


def test_encode_poly():
    """
    Testing encode_poly, output should be:
    [[ 1.  1.  1.]
     [ 1.  2.  4.]
     [ 1.  3.  9.]
     [ 1.  4. 16.]]
    """
    df = pd.DataFrame({'a': [1,2,3,4], 'z': [0,0,0,0]})
    aaa = encode_poly(df,'a',2)
    assert aaa.size == 12
    assert sum(aaa[0,:]) == 3
    assert aaa[3,2] == 16
    zzz = encode_poly(df,'z',3)
    assert zzz.size == 16
    assert sum(zzz[:,1]) == 0

    assert True
