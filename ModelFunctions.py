# Bayesian MMM
# Model Functions
# Nabeel Paruk
# 04/09/2024

from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM

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

