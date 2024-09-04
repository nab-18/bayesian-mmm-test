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
    channel_list = df_column_list.remove("date")
    channel_list = channel_list.remove("conversions")
    channel_list = channel_list.remove("trend")
    channel_list = channel_list.remove("year")
    channel_list = channel_list.remove("month")
    channel_list = channel_list.remove("dayofyear")

    total_spend_per_channel = df_features[[channel_list]].sum(axis=0)

    # Get spend proportion
    spend_proportion = total_spend_per_channel / total_spend_per_channel.sum()

    # Define parameters
    HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)
    n_channels = len(channel_list)

    # Prior sigma
    prior_sigma = HALFNORMAL_SCALE * n_channels * spend_proportion
    
    return prior_sigma, channel_list

def create_model_specification(df_features, channel_list):
    """
    Define feature and target variables and creates a dummy model specification using the default model configuration
    Args:
        df_features: DataFrame features returned from time_series_features function
        channel_list: Channel list retrieved from create_priors function
    """
    # Define feature and target variables
    X = df_features.drop("conversions", axis=1)
    y = df_features["conversions"]

    # Initialise the default model configuration
    dummy_model = DelayedSaturatedMMM(
        date_column = 'date',
        channel_columns = channel_list,
        control_columns = [
            "trend",
            "year",
            "month"
        ],
        adstock_max_lag=8
    )
    return dummy_model

def custom_config(int_dist='Normal', int_mu=0, int_sigma=2,
                  beta_dist='HalfNormal', beta_sigma=2,
                  ll_dist='Normal', ll_sigma_dist='HalfNormal', ll_sigma_sigma=2,
                  alpha_dist='Beta', alpha_alpha=1, alpha_beta=3,
                  lam_dist='Gamma', lam_alpha=1, lam_beta=1,
                  gam_con_dist='Normal', gam_con_mu=0, gam_con_sigma=2,
                  gam_fou_dist='Laplace', gam_fou_mu=0, gam_fou_b=1):
    # This needs more research. What changes does each parameter and distribution do and how can we communicate this in simpler terms?
    # Also need to allow for adjustment of the distributions and not just the parameters
    """
    Allow user to define a custom configuration for the model ( define custom priors)
    At this point, changing the distributions is not advised as the parameter names need to be adjusted in the function.
    Note that distributions can be specified on the hyperparameters.
    Args:
        int_*: The intercept specified distribution and hyperparameters
        beta_*: The beta channel specified distribution and hyperparameters
        ll_*: The likelihood specified distribution and hyperparameters
        alpha_*: The alpha specified distribution and hyperparameters
        lam_*: The lambda specified distribution and hyperparameters
        gam_con_*: The gamma control specified distribution and hyperparameters
        gam_fou_*: The gamma fourier specified distribution and hyperparameters
        *_dist: The specified distribution for the relevant parameter


    """
    my_model_config = {
        'intercept': {
            'dist': int_dist,
            'kwargs': {
                'mu': int_mu,
                'sigma': int_sigma
            }
        },
        'beta_channel': {
            'dist': beta_dist,
            'kwargs': {
                'sigma': beta_sigma
            }
        },
        "likelihood": {
            "dist": ll_dist,
            "kwargs":{
                "sigma":{
                    'dist': ll_sigma_dist,
                    'kwargs': {
                        'sigma': ll_sigma_sigma
                    }
                }
            }
        },
        'alpha': {
            'dist': alpha_dist,
            'kwargs': {'alpha': alpha_alpha, 'beta': alpha_beta}
        },
        'lam': {
            'dist': lam_dist,
            'kwargs': {'alpha': lam_alpha, 'beta': lam_beta}
        },
        'gamma_control': {
            'dist': gam_con_dist,
            'kwargs': {'mu': gam_con_mu, 'sigma': gam_con_sigma}
        },
        'gamma_fourier': {
            'dist': gam_fou_dist,
            'kwargs': {'mu': gam_fou_mu, 'b': gam_fou_b}
        },
    }

    return my_model_config

def delayed_saturated_mmm(my_model_config, channel_list, yearly_seasonality=2):
    """
    Creates sample configuration and defines your DelayedSaturatedMMM with the chosen parameters.
    Args:
        my_model_config: The model configuration chosen in custom_config
        channel_list: The list of channels in the dataset
        yearly_seasonality: The number of seasonal cycles that should be considered
    """
    # Define sample configuration
    my_sampler_config = {
        "progressbar": True,
        "cores": 1,
    }
    # Define the DelayedSaturatedMMM
    mmm = DelayedSaturatedMMM(
        model_config = my_model_config,
        sampler_config = my_sampler_config,
        date_column = "date",
        channel_columns = channel_list,
        control_columns = [
            "trend",
            "year",
            "month",
        ],
        adstock_max_lag=8,
        yearly_seasonality=yearly_seasonality
    )

    return mmm

def fit_model(model, X, y, random_seed=1):
    """
    Fit your model on the data.
    Args:
        X: Feature variables defined in create_model_specification
        y: Target variables created in create_model_specification
        target_accept: The 
    """
    model.fit(X, y, target_accept=0.95, random_seed=random_seed)


def main():
    """
    Main body of code. Create Streamlit implementation here.
    """
    # Headers and introduction
    st.title("Bayesian Marketing Mixed Model")
    st.write("This is your instructions -> Write something")

    # Upload file
    uploaded_file = st.file_uploader("Upload your csv file", type=['csv'])

    # Perform EDA

    # Preprocessing
    #  - Process for time series features
    df = read_in_data(uploaded_file)
    df_features, df_column_list = time_series_features(df)

    



if __name__ == '__main__':
    main()