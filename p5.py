"""
    Name: Jacqueline Saad
    Email: jacqueline.saad05@myhunter.cuny.edu
    Resources: Used pandas and sklearn docs.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering

def make_df(file_name):
    '''
    Reads the CSV file into a DataFrame.
    '''

    df = pd.read_csv(file_name)

    # Drop rows with null values in specific columns.
    df = df.dropna(subset=['TYP_DESC', 'INCIDENT_DATE', 'INCIDENT_TIME',
                           'Latitude', 'Longitude'])

    # Keep only rows with 'AMBULANCE' in the 'type_description' column.
    df = df[df['TYP_DESC'].str.contains('AMBULANCE', case=False)]

    return df

def add_date_time_features(df):
    '''
    Add date and time features to the given DataFrame.
    '''

    # Convert 'INCIDENT_DATE' to datetime format.
    df['INCIDENT_DATE'] = pd.to_datetime(df['INCIDENT_DATE'])

    # Add column 'WEEK_DAY' with day of the week
    # (0 for Monday, 1 for Tuesday, etc).
    df['WEEK_DAY'] = df['INCIDENT_DATE'].dt.dayofweek

    # Convert 'INCIDENT_TIME' to timedelta format.
    df['INCIDENT_TIME'] = pd.to_timedelta(df['INCIDENT_TIME'])

    # Add column 'INCIDENT_MIN' that represents
    # the number of minutes since midnight.
    df['INCIDENT_MIN'] = df['INCIDENT_TIME'].dt.total_seconds() / 60

    return df

def filter_by_time(df, days=None, start_min=0, end_min=1439):
    """
    Filter the given DataFrame based on specified days of the week and time range.
    """

    # Filter by days if specified.
    if days is not None:
        df = df[df['WEEK_DAY'].isin(days)]

    # Filter by incident times.
    df = df[(df['INCIDENT_MIN'] >= start_min) & (df['INCIDENT_MIN'] <= end_min)]

    return df

def compute_kmeans(df, num_clusters=8, n_init='auto', random_state=1870):
    """
    Run the KMeans model on Latitude and Longitude data of the provided DataFrame.
    """

    # Select Latitude and Longitude columns for clustering.
    coordinates = df[['Latitude', 'Longitude']]

    # Run KMeans model.
    kmeans = KMeans(n_clusters=num_clusters, n_init=n_init, random_state=random_state)
    kmeans.fit(coordinates)

    # Get cluster centers and predicted labels.
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return cluster_centers, labels

def compute_gmm(df, num_clusters=8, random_state=1870):
    """
    Run the GaussianMixture model on Latitude
    and Longitude data of the provided DataFrame.
    """

    # Select Latitude and Longitude columns for clustering.
    coordinates = df[['Latitude', 'Longitude']]

    # Run GaussianMixture model.
    gmm = GaussianMixture(n_components=num_clusters, random_state=random_state)
    labels = gmm.fit_predict(coordinates)

    return labels

def compute_agglom(df, num_clusters=8, linkage='ward'):
    """
    Run the AgglomerativeClustering model on Latitude
    and Longitude data of the provided DataFrame.
    """

    # Select Latitude and Longitude columns for clustering.
    coordinates = df[['Latitude', 'Longitude']]

    # Run AgglomerativeClustering model.
    agglom = AgglomerativeClustering(n_clusters=num_clusters, linkage=linkage)
    labels = agglom.fit_predict(coordinates)

    return labels

def compute_spectral(df, num_clusters=8, affinity='rbf', random_state=1870):
    """
    Run the SpectralClustering model on Latitude
    and Longitude data of the provided DataFrame.
    """

    # Select Latitude and Longitude columns for clustering.
    coordinates = df[['Latitude', 'Longitude']]

    # Run SpectralClustering model.
    spectral = SpectralClustering(n_clusters=num_clusters,
                                  affinity=affinity, random_state=random_state)
    labels = spectral.fit_predict(coordinates)

    return labels

def compute_explained_variance(df, k_vals=None, random_state=1870):
    """
    Compute the sum of squared distances of samples
    to their closest cluster center for each value of K.
    """

    if k_vals is None:
        k_vals = [1, 2, 3, 4, 5]

    # Select Latitude and Longitude columns for clustering.
    coordinates = df[['Latitude', 'Longitude']]

    # Compute explained variance for each K value.
    explained_variance = []

    for num_clusters in k_vals:
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        kmeans.fit(coordinates)
        explained_variance.append(kmeans.inertia_)

    return explained_variance

def test_add_date_time_features():
    """
    Test add_date_time_features function.
    """
    # Create a sample DataFrame.
    df = pd.DataFrame({
        'INCIDENT_DATE': ['2023-11-01', '2023-11-02', '2023-11-03'],
        'INCIDENT_TIME': ['12:30:00', '15:45:00', '20:10:00'],
        'Latitude': [40.1, 40.2, 40.3],
        'Longitude': [-73.9, -74.1, -73.8]
    })

    # Apply the function.
    df_result = add_date_time_features(df)

    # Check if new columns are present.
    assert 'WEEK_DAY' in df_result.columns
    assert 'INCIDENT_MIN' in df_result.columns

    # Check if data types are correct.
    assert df_result['WEEK_DAY'].dtype == np.dtype('int64')
    assert df_result['INCIDENT_MIN'].dtype == np.dtype('float64')

def test_filter_by_time():
    """
    Test filter_by_time function.
    """

    # Create a sample DataFrame.
    df = pd.DataFrame({
        'WEEK_DAY': [0, 1, 2, 3, 4],
        'INCIDENT_MIN': [720, 900, 1210, 1439, 600],
        'Latitude': [40.1, 40.2, 40.3, 40.4, 40.5],
        'Longitude': [-73.9, -74.1, -73.8, -73.7, -73.6]
    })

    # Apply the function.
    df_result = filter_by_time(df, days=[0, 1, 2, 3, 4], start_min=600, end_min=1200)

    # Check if the DataFrame is filtered correctly.
    assert len(df_result) == 4
    assert set(df_result['WEEK_DAY']) == {0, 1, 2, 3, 4}
    assert all(600 <= df_result['INCIDENT_MIN']) and all(df_result['INCIDENT_MIN'] <= 1200)

if __name__ == "__main__":
    # Add any additional code to run when executing the script directly.
    pass
