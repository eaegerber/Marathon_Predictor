
import numpy as np
import pandas as pd
from typing import Tuple
from utils import store_initial_prior, get_training_set, get_test_set, read_in_jsons, store_lk_json, store_processed_likelihoods


def bayes_iter(prior: pd.Series, likelihood: np.array) -> np.array:
    """Compute an iteration of a Bayesian update. Multiply the prior to the likelihood and return
    the posterior distribution as a numpy array"""
    unnorm = likelihood * prior  # unnorm = np.array(likelihood)  # * prior
    return unnorm / unnorm.sum()


def _prior_dist(informed: bool = True, max_time: int = 500) -> np.array:
    """Returns the prior distribution for the runner as a numpy array"""
    if informed:
        prior = np.loadtxt("processed_data/informed_prior.csv", delimiter=',')
    else:
        prior = np.ones(max_time) / max_time

    return prior


def full_bayes_dict(
        runner_info,
        initial_prior: np.array,
        likelihoods: dict,
) -> dict:
    """Compute the full bayes table for runner, doing each Bayesian update according to the runner info"""
    bayes_dict = {'0K': initial_prior}
    bayes_dict["Prior"] = bayes_dict["0K"]

    for dist, mark in runner_info:
        lk_table = likelihoods[dist].get(str(mark), np.ones(500))  # TODO instance of max_finish
        prior_array = bayes_dict["Prior"]
        bayes_dict[dist] = bayes_iter(prior=prior_array, likelihood=lk_table)
        bayes_dict["Prior"] = bayes_dict[dist]

    bayes_dict["Posterior"] = bayes_dict[runner_info[-1][0]]

    bayes_dict.pop("Prior")
    return bayes_dict


def person_dict(
        person: pd.Series,
        checkpoints: list,
        prior: np.array,
        lk_tables: dict,
) -> Tuple[dict, int]:
    """Return both bayes table and actual value"""
    actual = person["Finish Net"]
    bayes_dict = full_bayes_dict(
        runner_info=[[dist, person[dist]] for dist in checkpoints],
        initial_prior=prior,
        likelihoods=lk_tables,
    )
    return bayes_dict, actual


if __name__ == '__main__':
    data = get_training_set(pd.read_csv("processed_data/full_data_mins.csv"))
    store_initial_prior(data=data)
    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]

    store_lk_json(data=data, dists=marks)
    store_processed_likelihoods(max_length=500)
    lks = read_in_jsons(marks)

    people = get_test_set(pd.read_csv("processed_data/full_data_mins.csv"))
    show = marks

    max_finish = 500
    for i in range(len(people[:3000])):
        person_info = people.iloc[i]
        informed_prior = _prior_dist(informed=True, max_time=max_finish)
        dict1 = person_dict(person=person_info, checkpoints=marks, prior=informed_prior, lk_tables=lks)[0]

        uninformed_prior = _prior_dist(informed=False, max_time=max_finish)
        dict2 = person_dict(person=person_info, checkpoints=marks, prior=uninformed_prior, lk_tables=lks)[0]

        if i % 1000 == 0:
            print(i)
