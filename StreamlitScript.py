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
import time

# Functions
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
    df_features_column_list = df_features.columns.to_list()

    df_features = df_features[df_features_column_list]
    return df_features, df_column_list

def create_priors(df_features, df_column_list):
    """
    Setup the model and create the priors using the features
    Args:
        df_column_list: Column list returned from time_series_features function
        df_features: DataFrame features returned from time_series_features function
    """
    # Determine total spend per channel
    channel_list = df_column_list
    channel_list.remove("date")
    channel_list.remove("conversions")

    total_spend_per_channel = df_features[channel_list].sum(axis=0)

    # Get spend proportion
    spend_proportion = total_spend_per_channel / total_spend_per_channel.sum()

    # Define parameters
    HALFNORMAL_SCALE = 1 / np.sqrt(1 - 2 / np.pi)
    n_channels = len(channel_list)

    # Prior sigma
    prior_sigma = HALFNORMAL_SCALE * n_channels * spend_proportion
    
    return prior_sigma, channel_list

def create_model_specification(df_features, channel_list, yearly_seasonality=2):
    """
    Define feature and target variables and creates a dummy model specification using the default model configuration
    Args:
        df_features: DataFrame features returned from time_series_features function
        channel_list: Channel list retrieved from create_priors function
    """
    # Define feature and target variables
    X = df_features.drop("conversions", axis=1)
    y = df_features["conversions"]

    # Define sampler configuration
    my_sampler_config = {
        "progressbar": True,
        "cores": 1,
    }

    # Initialise the default model configuration
    dummy_model = DelayedSaturatedMMM(
        sampler_config=my_sampler_config,
        date_column = 'date',
        channel_columns = channel_list,
        control_columns = [
            "trend",
            "year",
            "month"
        ],
        adstock_max_lag=8,
        yearly_seasonality=yearly_seasonality
    )
    return dummy_model, X, y

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
        random_seed: Specify for reproducibility
    """
    model.fit(X, y, target_accept=0.95, random_seed=random_seed)
    return model


def main():
    """
    Main body of code. Create Streamlit implementation here.
    """
    # Headers and introduction
    st.title("Bayesian Marketing Mixed Model")
    st.write("This is your instructions -> Write something")

    # Upload file
    uploaded_file = st.file_uploader("Upload your csv file", type=['csv'])


    if uploaded_file is not None:
        # Perform EDA


        # Preprocessing
        #  - Process for time series features
        df = read_in_data(uploaded_file)
        df_features, df_column_list = time_series_features(df)

        # - Create the priors
        prior_sigma, channel_list = create_priors(df_features, df_column_list)

        # - Create model specification with default config
        dummy_model, X, y = create_model_specification(df_features, channel_list)

        # - Custom configuration?
            # We can create multiple premade configurations for different applications. More research is required
        specify_config = st.radio("Which configuration would you like to use?",
                                ["Default", "Custom"],
                                captions=[
                                    "Use the default configuration",
                                    "Create your own custom configuration. Leaving variables blank will result in default values."
                                ])
        if specify_config == "Default":
            # Set the final model to the default configuration
            final_model = dummy_model
            yearly_seasonality = st.text_input("Yearly seasonality (Defaults to 2)", 2)

        elif specify_config == "Custom":
            # Allow the user to input their custom configuration
            # - Intercept
            int_dist = 'Normal'
            int_mu = int(st.text_input("Intercept Normal Distribution mu parameter", 0))
            int_sigma = int(st.text_input("Intercept Normal Distribution sigma parameter", 2))

            # - Beta channel
            beta_dist='HalfNormal'
            beta_sigma = int(st.text_input("Beta_Channel HalfNormal Distribution sigma parameter", 2))

            # - Likelihood
            ll_dist='Normal'
            ll_sigma_dist = 'HalfNormal'
            ll_sigma_sigma = int(st.text_input("Likelihood Normal Distribution sigma hyperparameter", 2))

            # - Alpha
            alpha_dist='Beta'
            alpha_alpha = int(st.text_input("Alpha Beta Distribution alpha parameter", 1))
            alpha_beta = int(st.text_input("Alpha Beta Distribution beta parameter", 3))

            # - Lambda
            lam_dist='Gamma'
            lam_alpha = int(st.text_input("Lambda Gamma Distribution alpha parameter", 1))
            lam_beta = int(st.text_input("Lambda Gamma Distribution beta parameter", 1))

            # - Gamma control
            gam_con_dist='Normal'
            gam_con_mu = int(st.text_input("Gamma Control Normal Distribution mu parameter", 0))
            gam_con_sigma = int(st.text_input("Gamma Control Normal Distribution sigma parameter", 2))

            # - Gamma fourier
            gam_fou_dist='Laplace'
            gam_fou_mu = int(st.text_input("Gamma Fourier Laplace Distribution mu parameter", 0))
            gam_fou_b = int(st.text_input("Gamma Fourier Laplace Distribution beta parameter", 1))

            # Define configuration
            my_model_config = custom_config(int_dist, int_mu, int_sigma,
                    beta_dist, beta_sigma,
                    ll_dist, ll_sigma_dist, ll_sigma_sigma,
                    alpha_dist, alpha_alpha, alpha_beta,
                    lam_dist, lam_alpha, lam_beta,
                    gam_con_dist, gam_con_mu, gam_con_sigma,
                    gam_fou_dist, gam_fou_mu, gam_fou_b)
            
            # Create custom configuration model
            yearly_seasonality = st.text_input("Yearly seasonality (Defaults to 2)", 2)
            final_model = delayed_saturated_mmm(my_model_config, channel_list, yearly_seasonality)

        # Fit the model
        if st.button("Create model"):
            with st.status("Fitting model...") as status:
                final_model.fit(X, y, target_accept = 0.95, random_seed = 888)
                status.update(label="Model has been fit!", state="complete", expanded=False)
            

    
        # Visualisations
            if st.button("Evaluate the model"):  # Add descriptions to each plot, this is currently very uninformative and difficult to interpret
                #try:
                st.header("Visualisations")

                # Component contributions
                st.subheader("Posterior Predictive Model Components")
                fig = final_model.plot_components_contributions()
                st.pyplot(fig)

                # Direct contribution curves
                st.subheader("Direct Contribution Curves per Platform")
                fig = final_model.plot_direct_contribution_curves()
                st.pyplot(fig)

                # Channel contributions
                st.subheader("Channel Contributions as a function of cost share")
                fig = final_model.plot_channel_contributions_grid(start=0, stop=1.5, num=12, absolute_xrange=True)
                st.pyplot(fig)

                # Get ROAS
                # - Get the samples
                get_mean_contributions_over_time_df = final_model.compute_mean_contributions_over_time(original_scale=True)
                channel_contribution_original_scale = final_model.compute_channel_contribution_original_scale()

                roas_samples = (
                    channel_contribution_original_scale.stack(sample=("chain", "draw")).sum("date")
                    / X[channel_list].sum().to_numpy()[..., None]
                )

                # - Visualise the samples
                st.subheader("Visualise the ROAS Posterior Distribution")
                fig, ax = plt.subplots(figsize=(15, 6))
                for channel in channel_list:
                    sns.histplot(
                        roas_samples.sel(channel=channel).to_numpy(), binrange=(0,0.005) ,alpha=0.3, kde=True, ax=ax, legend=True, label = channel
                    )
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
                ax.set(title="Posterior ROAS distribution", xlabel="ROAS")
                st.pyplot(fig)

                # - Get the ROAS summary per platform in a dataframe
                st.subheader("ROAS Summary per Platform")
                roas_df = roas_samples.to_dataframe(name="roas")

                roas_df.groupby("channel").mean()
                roas_summary = roas_df.groupby("channel")['roas'].describe(percentiles=[0.025, 0.975])
                st.dataframe(roas_summary)
            
            #except:
                # st.warning("Your model might not have been fit. Please fit the model before evaluating it.")


if __name__ == '__main__':
    main()