# Bayesian MMM
# Main
# 04/09/2024
# Nabeel Paruk

import streamlit as st
import pandas as pd
import numpy as np
import pytime as tk
from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM
import seaborn as sns
import matplotlib.pyplot as plt
from ModelFunctions import *
from PreprocessFunctions import *

def main():
    """
    Main body of code. Create Streamlit implementation here.
    """
    # Headers and introduction
    st.title("Bayesian Marketing Mixed Model")
    st.write("## Data upload format")
    st.write("1. Upload your data in csv format")
    st.write("2. Column 1: Labelled 'Date' and have the format YYYY/MM/DD.")
    st.write("3. Column 2: This should be your target variable (e.g. conversions, impressions, clicks etc).")
    st.write("4. Columns 3-N: These are your channels and the associated spend per date.")

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
        if st.button("Create model", key=0):
            with st.status("Fitting model...") as status:
                final_model.fit(X, y, target_accept = 0.95, random_seed = 888)
                status.update(label="Model has been fit!", state="complete", expanded=False)
            

    
                # Visualisations
                #if st.button("Evaluate the model", key=1):  # Add descriptions to each plot, this is currently very uninformative and difficult to interpret
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