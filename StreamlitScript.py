# Bayesian MMM Test
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
    return df_features()




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