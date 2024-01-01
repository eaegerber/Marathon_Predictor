
import numpy as np
import pandas as pd
from typing import Tuple
from likelihoods import main_lk
from utils import store_initial_prior, get_training_set, get_test_set


def bayes_iter(prior: pd.Series, likelihood: np.array) -> np.array:
    """Compute an iteration of a Bayesian update. Multiply the prior to the likelihood and return
    the posterior distribution as a numpy array"""
    unnorm = likelihood * prior
    a = unnorm / unnorm.sum()
    smoothing = 0.00001
    return (a + smoothing) / (a + smoothing).sum()


def _prior_dist(informed: bool = True, max_time: int = 500) -> np.array:
    """Returns the prior distribution for the runner as a numpy array"""
    if informed:
        prior = np.loadtxt("processed_data/informed_prior.csv", delimiter=',')
    else:
        prior = np.ones(max_time) / max_time

    return prior


def full_bayes_dict(
        runner_dict: dict,
        initial_prior: np.array,
        likelihoods: dict,
) -> dict:
    """Compute the full bayes table for runner, doing each Bayesian update according to the runner info"""
    runner_dict["0K"] = 0
    last_dict = {"5K": "0K", "10K": "5K", "15K": "10K", "20K": "15K", "HALF": "20K",
                 "25K": "HALF", "30K": "25K", "35K": "30K", "40K": "35K"}
    bayes_dict = {'0K': initial_prior}
    bayes_dict["Prior"] = bayes_dict["0K"]

    for dist in ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]:
        last_dist = last_dict[dist]
        last_mark = str(float(runner_dict[last_dist]))
        curr_mark = str(float(runner_dict[dist]))

        last_lk = likelihoods[dist].get(str(last_mark), np.ones(1001))
        lk_array = last_lk.get(str(curr_mark), np.ones(1001))
        p1, p2 = int(lk_array[0]), np.array(lk_array[1:])
        lk_array = np.concatenate([np.zeros(p1), p2])
        n = len(lk_array)
        lk_array = np.concatenate([lk_array, np.zeros(1001 - n)])
        prior_array = bayes_dict["Prior"]
        bayes_dict[dist] = bayes_iter(prior=prior_array, likelihood=lk_array)
        bayes_dict["Prior"] = bayes_dict[dist]

    # bayes_dict["Posterior"] = bayes_dict[runner_info[-1][0]]

    bayes_dict.pop("Prior")
    bayes_dict = {k: v[::2] for k, v in bayes_dict.items()}
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
        runner_dict={dist: person[dist] for dist in checkpoints},
        initial_prior=prior,
        likelihoods=lk_tables,
    )
    return bayes_dict, actual


if __name__ == '__main__':
    max_finish = 1001
    data = get_training_set(pd.read_csv("processed_data/full_data_secs.csv"))
    store_initial_prior(data=data, max_time=max_finish)
    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    lks = main_lk(df=data, marks_list=marks, store=False, process=False)
    people = get_test_set(pd.read_csv("processed_data/full_data_mins.csv"))

    for i in range(len(people[:3001])):
        person_info = people.iloc[i]
        informed_prior = _prior_dist(informed=True, max_time=max_finish)
        dict1 = person_dict(person=person_info, checkpoints=marks, prior=informed_prior, lk_tables=lks)[0]

        uninformed_prior = _prior_dist(informed=False, max_time=max_finish)
        dict2 = person_dict(person=person_info, checkpoints=marks, prior=uninformed_prior, lk_tables=lks)[0]

        if i % 1000 == 0:
            print(i)
