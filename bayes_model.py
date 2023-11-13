
import numpy as np
import pandas as pd
from typing import Tuple
from utils import store_initial_prior, get_test_set_people  # store_likelihood_tables
# from collections import defaultdict
# from visualize import prior_compare


def bayes_iter(
        prior_array: pd.Series,
        lk_table: pd.DataFrame,
        mark: int,
) -> np.array:
    """Compute an iteration of a Bayesian update. Multiply the prior to the likelihood and return
    the posterior distribution as a numpy array"""
    prior = prior_array
    likelihood = lk_table[str(mark)].to_numpy()
    unnorm = prior * likelihood
    return unnorm / unnorm.sum()


def _prior_dist(informed: bool = True, max_time: int = 500) -> np.array:
    """Returns the prior distribution for the runner as a numpy array"""
    if informed:
        prior = np.loadtxt("processed_data/informed_prior.csv", delimiter=',')
    else:
        prior = np.ones(max_time + 1) / max_time + 1

    return prior  # np.array(pd.DataFrame(prior, index=range(max_time + 1)).fillna(0)).reshape(max_time + 1)  # array


def full_bayes_table(
        runner_info,
        initial_prior: np.array,
        likelihoods: dict,
):
    """Compute the full bayes table for runner, doing each Bayesian update according to the runner info"""
    bayes_dict = {'0K': initial_prior}
    bayes_dict["Prior"] = bayes_dict["0K"]

    for dist, mark in runner_info:
        lk_table = likelihoods[dist]
        prior_array = bayes_dict["Prior"]
        bayes_dict[dist] = bayes_iter(prior_array=prior_array, lk_table=lk_table, mark=mark)
        bayes_dict["Prior"] = bayes_dict[dist]

    bayes_dict["Posterior"] = bayes_dict[runner_info[-1][0]]

    bayes_dict.pop("Prior")
    bayes_table = pd.DataFrame(bayes_dict)
    return bayes_table


def person_table(
        person: pd.Series,
        checkpoints: list,
        prior: np.array,
        lk_tables: dict,
) -> Tuple[pd.DataFrame, int]:
    """Return both bayes table and actual value"""
    actual = person["Finish Net"]
    return full_bayes_table(
        runner_info=[[dist, person[dist]] for dist in checkpoints],
        initial_prior=prior,
        likelihoods=lk_tables,
    ), actual


if __name__ == '__main__':
    store_initial_prior(data=pd.read_csv("processed_data/full_data_mins.csv"))

    # people = pd.read_csv("processed_data/nucr_runners.csv", index_col=0)
    people = get_test_set_people(pd.read_csv("processed_data/full_data_mins.csv"))
    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    show = marks

    max_finish = 500
    # df = pd.read_csv("processed_data/full_data_mins.csv")
    # df = df[df["Year"] != 2023]
    # store_likelihood_tables(data=df, marks=checkpoints, finish_range=max_finish, mark_range=max_finish)

    lk_dict = {mark: pd.read_csv(f"likelihood_tables/likelihood_{mark}.csv") for mark in marks}

    for i in range(len(people[:1000])):
        person_info = people.iloc[i]
        informed_prior = _prior_dist(informed=True, max_time=max_finish)
        table1 = person_table(person=person_info, checkpoints=marks, prior=informed_prior, lk_tables=lk_dict)[0]

        uninformed_prior = _prior_dist(informed=False, max_time=max_finish)
        table2 = person_table(person=person_info, checkpoints=marks, prior=uninformed_prior, lk_tables=lk_dict)[0]

        print(f"Finished: {people.iloc[i]['Name']}")
