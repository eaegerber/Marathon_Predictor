
from pymc_experimental.model_builder import ModelBuilder
from typing import Dict, Union
import numpy as np
import arviz as az
import pandas as pd
import pymc as pm

marks = ['_', "5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]

class LinearModel(ModelBuilder):
    # Give the model a name
    _model_type = "LinearModel"

    # And a version
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """
        # Check the type of X and y and adjust access accordingly
        X_values = X # ["input"].values
        y_values = y.values if isinstance(y, pd.Series) else y
        self._generate_and_preprocess_model_data(X_values, y_values)

        with pm.Model(coords=self.model_coords) as self.model:
            # Create mutable data containers
            x_data = pm.Data("x_data", X_values["total_pace"], dims="obs_id", mutable=True)
            y_data = pm.Data("y_data", y_values, dims="obs_id", mutable=True)

            # prior parameters
            b0_mu_prior = self.model_config.get("b0_mu_prior", 0)
            b0_sigma_prior = self.model_config.get("b0_sigma_prior", 5)
            b1_mu_prior = self.model_config.get("b1_mu_prior", 0)
            b1_sigma_prior = self.model_config.get("b1_sigma_prior", 5)
            sigma_beta_prior = self.model_config.get("sigma_beta_prior", 2) 

            # priors
            b_0 = pm.Normal("b_0", mu=b0_mu_prior, sigma=b0_sigma_prior, dims="group")
            b_1 = pm.Normal("b_1", mu=b1_mu_prior, sigma=b1_sigma_prior, dims="group")
            
            categories = np.array(marks[1:]) 
            dist_idx = pd.Categorical(X_values["dist"], categories=categories).codes
            g = pm.Data("g", dist_idx, dims="obs_id", mutable=True)
            sigma = pm.HalfCauchy("sigma", beta=sigma_beta_prior, dims="group")

            obs = pm.Normal("y", mu=b_0[g] + b_1[g] * x_data, sigma=sigma[g], shape=x_data.shape, observed=y_data, dims="obs_id")

    def _data_setter(
        self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray] = None
    ):
        if isinstance(X, pd.DataFrame):

            # NEW
            categories = np.array(marks[1:]) 
            dist_idx = pd.Categorical(X["dist"], categories=categories).codes

            x_values = X["total_pace"] #["total_pace"]#X   #["input"].values
        else:
            # Assuming "input" is the first column
            x_values = X[:, 0]

        # self._generate_and_preprocess_model_data(x_values, y)  # NEW
        with self.model:
            pm.set_data({"x_data": x_values, "g": dist_idx}, coords=self.model_coords)
            if y is not None:
                pm.set_data({"y_data": y.values if isinstance(y, pd.Series) else y})

    @staticmethod
    def default_model_config() -> Dict:
        """
        Returns a class default config dict for model builder if no model_config is provided on class initialization.
        The model config dict is generally used to specify the prior values we want to build the model with.
        It supports more complex data structures like lists, dictionaries, etc.
        It will be passed to the class instance on initialization, in case the user doesn't provide any model_config of their own.
        """
        model_config: Dict = {
            "b0_mu_prior": 0, 
            "b0_sigma_prior": 5, 
            "b1_mu_prior": 0, 
            "b1_sigma_prior": 5, 
            "sigma_beta_prior": 2, 
        }
        return model_config

    @staticmethod
    def default_sampler_config() -> Dict:
        """
        Returns a class default sampler dict for model builder if no sampler_config is provided on class initialization.
        The sampler config dict is used to send parameters to the sampler .
        It will be used during fitting in case the user doesn't provide any sampler_config of their own.
        """
        sampler_config: Dict = {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 4,
            "target_accept": 0.95,
            "idata_kwargs": {'log_likelihood':True},
        }
        return sampler_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """
        pass

    def _generate_and_preprocess_model_data(
        self, X: Union[pd.DataFrame, pd.Series], y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Depending on the model, we might need to preprocess the data before fitting the model.
        all required preprocessing and conditional assignments should be defined here.
        """
        # self.model_coords = None  
        group_list = marks[1:]
        self.model_coords = {"group": group_list}
        # in our case we're not using coords, but if we were, we would define them here, or later on in the function, if extracting them from the data.
        # as we don't do any data preprocessing, we just assign the data given by the user. Note that it's a very basic model,
        # and usually we would need to do some preprocessing, or generate the coords from the data.
        self.X = X
        self.y = y


    # New
    def prediction(self, X_test: Union[pd.DataFrame, np.ndarray], trace, progressbar=True):
        self._data_setter(X_test)
        post_pred = pm.sample_posterior_predictive(trace, self.model, predictions=True, progressbar=progressbar)
        return (42195 / 60) / az.extract(post_pred.predictions)['y'].data