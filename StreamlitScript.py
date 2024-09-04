# Bayesian MMM Test
# Nabeel Paruk
# 04/09/2024
# Notes:
# - 'conversions' needs to be adjusted as a name and be allowed to vary i.e. to impressions or ctr etc
# - The EDA section needs to be completed
# - 

import streamlit as st
import pandas as pd
import numpy as np
import pytime as tk
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM
import seaborn as sns
import matplotlib.pyplot as plt

def read_in_data(upload):
    """
    Reads in data and preprocesses it for EDA
    Args:
        upload: csv file to be processed
    """
    # Read in data and parse dates
    data = pd.read_csv(upload, parse_dates=['Date'])
    df = data.copy()

    # Make column names lower case
    df.columns = [col.lower() for col in df.columns]

    return df

def time_series_features(df):
    """
    Creates the time series features required for the model
    Args:
        df: DataFrame that has already been processed in read_in_data function
    """
    # Create the time series features
    df_features = df \
        .assign(
            year = lambda x: x["date"].dt.year,
            month = lambda x: x["date"].dt.month,
            dayofyear = lambda x: x["date"].dt.dayofyear,
        ) \
        .assign(
            trend = lambda x: df.index,
        )
    
    # Make column names lower case and add to a list
    df.columns = [col.lower() for col in df.columns]
    df_column_list = df.columns.to_list()

    df_features = df_features[[df_column_list]]
    return df_features, df_column_list

def create_priors(df_features, df_column_list):
    """
    Setup the model and create the priors using the features
    Args:
        df_column_list: Column list returned from time_series_features function
        df_features: DataFrame features returned from time_series_features function
    """
    # Determine total spend per channel
    df_temp_col_list = df_column_list.remove("date")
    df_temp_col_list = df_temp_col_list.remove("conversions")
    df_temp_col_list = df_temp_col_list.remove("trend")
    df_temp_col_list = df_temp_col_list.remove("year")
    df_temp_col_list = df_temp_col_list.remove("month")
    df_temp_col_list = df_temp_col_list.remove("dayofyear")

    total_spend_per_channel = df_features[[df_temp_col_list]].sum(axis=0)

    # Get spend proportion
    spend_proportion = total_spend_per_channel / total_spend_per_channel.sum()

    # Define parameters
    HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)
    n_channels = len(df_temp_col_list)

    # Prior sigma
    prior_sigma = HALFNORMAL_SCALE * n_channels * spend_proportion
    
    return prior_sigma

def create_model_specification(df_features):
    """
    Define feature and target variables and create the model specification using the default model configuration
    Args:
        df_features: DataFrame features returned from time_series_features function
    """
    # Define features and target variables
    X = 


def main():
    # Headers and introduction
    st.title("Bayesian Marketing Mixed Model")
    st.write("This is your instructions -> Write something")

    # Upload file
    uploaded_file = st.file_uploader("Upload your csv file", type=['csv'])

    # Perform EDA

    # Process for time series features
    df = read_in_data(uploaded_file)




if __name__ == '__main__':
    main()