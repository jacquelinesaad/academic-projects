"""
Katherine St. John
katherine.stjohn@hunter.cuny.edu
Program 2, Fall 2023
Resources:  Used Pandas docs for syntax
"""
import math
import pandas as pd



def import_data(file_name):
    """
    The data in the file is read into a DataFrame, and the columns:
    TBA are kept.
    Any rows with non-positive total_amount are dropped.
    The resulting DataFrame is returned.
    """

    df = pd.read_csv(file_name)
    df = df[ ['dbn','school_name','NTA','graduation_rate','pct_stu_safe',\
            'attendance_rate','college_career_rate','language_classes',\
            'advancedplacement_courses','method1','overview_paragraph'] ]
    df = df[ df['graduation_rate'].notnull()]
    return df

def impute_numeric_cols(df):
    """
    Missing entries in the numeric columns are replaced with the median
    of the respective column.
    Returns the resulting DataFrame.
    """
    num_cols = ['pct_stu_safe','attendance_rate', 'college_career_rate']
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    return df

def compute_count_col(df,col):
    """
    Counts the number of comma separated items in each entry of df[col].
    Returns the resulting Series.
    """

    return df[col].apply(lambda val: val.count(',')+1 if isinstance(val,str) else 0)

def encode_categorical_col(col):
    """
    One hot encoding of col, using col's values as labels.
    Returns resulting df.
    """
    df = col.str.get_dummies(sep=', ')
    df.sort_index(axis=1, inplace=True)
    return df

def split_test_train(df, xes_col_names, y_col_name, frac=0.25, random_state=922):
    """
    Splits the df into testing and training sets.
    Returns the results.
    """
    df_test = df.sample(frac=frac,random_state=random_state)
    df_train = df.copy()
    df_train = df_train.drop(df_test.index)
    x_train = df_train[xes_col_names]
    x_test = df_test[xes_col_names]
    y_train = df_train[y_col_name]
    y_test = df_test[y_col_name]
    return x_train, x_test, y_train, y_test

def compute_lin_reg(xes,yes):
    """
    The function computes the slope and y-intercept of the
    linear regression line, using ordinary least squares.
    Returns the y-intecept, theta_0, and the slope, theta_1
    """
    theta_1 = xes.corr(yes)*yes.std()/xes.std()
    theta_0 = yes.mean() - theta_1*xes.mean()
    return theta_0, theta_1

def predict(xes, theta_0, theta_1):
    """
    The function returns the predicted values of the dependent
    variable, xes, under the linear regression model with
    slope, theta_1 and y_intercept, theta_0.
    """
    return xes*theta_1 + theta_0

def mse_loss(y_actual,y_estimate):
    """
    Returns the mean squared error loss of the two series.
    """
    return sum( (y_actual-y_estimate)**2 )/len(y_actual)

def rmse_loss(y_actual,y_estimate):
    """
    Returns the mean squared error loss of the two series.
    """
    return( math.sqrt( mse_loss(y_actual,y_estimate) ))

def compute_error(y_actual,y_estimate,loss_fnc=mse_loss):
    """
    Computes the error for the two series, using the specified loss_fnc.
    """
    return loss_fnc(y_actual,y_estimate)

def compute_mse(theta, counts):
    """
    Computes the Mean Squared Error of the parameter theta and a dictionary, counts.
    Returns the MSE.
    """
    mse = 0

    num_values = sum(counts.values())
    total = sum([((k-theta)**2)*v for k,v in counts.items()])

    mse = total/num_values

    return mse

def test_compute_count_col(compute_fnc=compute_count_col):
    """
    Returns True if the compute_fnc performs correctly
    (e.g. computes counts) and False otherwise.
    """

    #Set up a df for testing:
    df = pd.DataFrame({'Letters': ['A,B','A,B'],
                       'Empty' : ["",""],
                       'AP' : ["Bio,CS,US","Bio"]
                       })
    test0 = compute_fnc(df,'Empty')
    test1 = compute_fnc(df,'Letters')
    test2 = compute_fnc(df,'AP')

    if test0[0] != 2:
        correct = False
    elif test0[1] != 2:
        correct = False
    if test1[0] != 0:
        correct = False
    elif test1[1] != 0:
        correct = False
    if test2[0] != 3:
        correct = False
    elif test2[1] != 1:
        correct = False
    else:
        correct = True

    return correct


def test_predict(predict_fnc=predict):
    """
    Returns True if the predict_fnc performs correctly
    (e.g. predicts values) and False otherwise.
    """

    #Should return theta_0 since theta_0 + 0*theta_1 = theta_0
    xes0 = pd.Series([0,0,0,0])
    #Should return theta_0 + theta_1
    xes1 = pd.Series([1,1,1,1])
    #Should return theta0+i*theta1
    xes2 = pd.Series([1,2,3,4,5,6])

    print(sum (predict_fnc(xes0,1,0)-(xes1)))
    print(sum (predict_fnc(xes0,0,0)))
    print(sum (predict_fnc(xes0,5,5) - (5*xes1)))
    if sum (abs(predict_fnc(xes0,1,0)-(xes1))) > 0.01:
        correct = False
    elif sum (abs(predict_fnc(xes0,0,0))) > 0.01:
        correct = False
    elif sum (abs(predict_fnc(xes0,5,5) - (5*xes1))) > 0.01:
        correct = False
    elif sum( abs(predict_fnc(xes2,0,0)-(xes0))) > 0.01:
        correct = False
    else:
        correct = True

    return correct

def test_mse_loss(error_fnc=mse_loss):
    """
    Returns True if the error_fnc performs correctly
    (e.g. computes MSE) and False otherwise.
    """

    #Some test vectors:
    xes0 = pd.Series([0,0,0,0])
    xes1 = pd.Series([1,1,1,1])
    xes2 = pd.Series([1,2,3,4])


    if error_fnc(xes0,xes0) != 0:
        correct = False
    elif error_fnc(xes2,xes2) != 0:
        correct = False
    elif error_fnc(xes0,xes1) != 1:
        correct = False
    elif error_fnc(xes0,5*xes1) != 25:
        correct = False
    else:
        correct = True

    return correct


def main():
    """
    Some examples of the functions in use:
    """
    ###Extracts the overviews from the data files:
    file_name = 'fall23/program02/2021_DOE_High_School_Directory_SI.csv'
    df_si = import_data(file_name)
    print(f'There are {len(df_si.columns)} columns:')
    print(df_si.columns)
    print('The dataframe is:')
    print(df_si)

    file_name = 'fall23/program02/2020_DOE_High_School_Directory_late_start.csv'
    df_late = import_data(file_name)
    print('The numerical columns are:')
    print(df_late[ ['dbn','pct_stu_safe','attendance_rate','college_career_rate'] ])

    ###Impute values, using median, for the numeric columns:
    df_late = impute_numeric_cols(df_late)
    print(df_late[ ['dbn','pct_stu_safe','attendance_rate','college_career_rate'] ])

    ###Count number of languages & AP courses offered:
    df_si['language_count'] = compute_count_col(df_si,'language_classes')
    df_si['ap_count'] = compute_count_col(df_si,'advancedplacement_courses')
    print('Staten Island High Schools:')
    print(df_si[ ['dbn','language_count','language_classes','ap_count',\
                  'advancedplacement_courses'] ])

    df_late['language_count'] = compute_count_col(df_late,'language_classes')
    df_late['ap_count'] = compute_count_col(df_late,'advancedplacement_courses')
    print('High schools that have 9am or later start:')
    print(df_late[ ['dbn','language_count','language_classes','ap_count',\
                    'advancedplacement_courses'] ])

    ###One hot encoding by language:
    df_langs = encode_categorical_col(df_si['language_classes'])
    print(df_langs)
    print('Number of schools for each language:')
    print(df_langs.sum(axis=0))


    ###Splitting data into training and test sets:
    xes_cols = ['language_count','ap_count','pct_stu_safe','attendance_rate','college_career_rate']
    y_col = 'graduation_rate'
    x_train, x_test, y_train, y_test = split_test_train(df_late,xes_cols,y_col)
    print('The sizes of the sets are:')
    print(f'x_train has {len(x_train)} rows.\tx_test has {len(x_test)} rows.')
    print(f'y_train has {len(y_train)} rows.\ty_test has {len(y_test)} rows.')

    ###Build models for graduation_rate on training data:
    coeff = {}
    for col in xes_cols:
        coe = compute_lin_reg(x_train[col],y_train)
        coeff[col] = coe        
        print(f'for {col}, theta_0 = {coe[0]} and theta_1 = {coe[1]}')

    ###Compute errors on training data:
    predicts = {}
    errors = {}
    min_error = 1
    best = ""
    for col in xes_cols:
        predicts[col] = predict(x_test[col], coe[0], coe[1])
        errors[col] = compute_error(y_test,predicts[col])
        print(f'Error on test data for {col} is {errors[col]}.')
        if errors[col] < min_error:
            min_error = errors[col]
            best = col

    print(f'Column {col} has lowest error ({min_error}).')

    ###Graphing actual and predicted for the models:
    import matplotlib.pyplot as plt
    import seaborn as sns
    def graph_data(df, col, coeff):
        """
        Function to graph the models
        """
        plt.scatter(df[col],df['graduation_rate'],label='Actual')
        predict_grad = predict(df_late[col],coeff[col][0],coeff[col][1])
        plt.scatter(df[col], predict_grad,label='Predicted')
        plt.title(f'{col} vs graduation_rate')
        plt.ylabel('graduation_rate')
        plt.xlabel(f'{col}')
        plt.legend()
        plt.show()

    graph_data(df_late, 'college_career_rate',coeff)

    ###Testing
    #Trying first on the correct function:
    print(f'test_compute_count_col(compute_count_col) returns\
           {test_compute_count_col(compute_count_col)}.')


if __name__ == "__main__":
    main()
