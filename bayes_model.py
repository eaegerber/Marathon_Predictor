
import numpy as np
import pandas as pd
from typing import Tuple
from likelihoods import get_likelihood, get_s2_dict
from utils import store_initial_prior, get_train_set, get_test_set, _prior_dist, _get_marks


def bayes_iter(prior: np.array, lk: np.array, smoothing_factor: float = 0.00001) -> np.array:
    """Compute an iteration of a Bayesian update. Multiply the prior array
    to the likelihood array, normalize, add a smoothing factor to each value,
    normalize again, and return the posterior distribution as a numpy array"""
    unnorm = lk * prior
    if unnorm.sum() == 0:
        normed = prior
    else:
        normed = unnorm / unnorm.sum()
    return (normed + smoothing_factor) / (normed + smoothing_factor).sum()


def full_bayes_dict(runner_dict: dict, initial_prior: np.array, lk_data: pd.DataFrame, s2) -> dict:
    """Compute the full bayes table for a runner, doing each Bayesian update
     according to the {dist: mark} pairings in the `runner info` dict
     :param runner_dict: {dict: mark} pairings for the runner
     :param initial_prior: array for the initial prior distribution
     :param lk_data: lk data # TODO fix
     :param s2: precomputed s2 data # TODO fix
     """
    bayes_dict = {'0K': initial_prior}
    bayes_dict["Prior"] = bayes_dict["0K"]

    person_marks = _get_marks(marks_list=list(runner_dict.keys()), zero_k=True, finish=True)  # adds 0k if needed
    last_dict = dict(zip(person_marks[1:], person_marks))  # {"5K": "0K", "10K": "5K", ...}
    col_map = {dist: num for num, dist in enumerate(_get_marks(marks_list=None, zero_k=True, finish=True))}
    for dist in person_marks:
        if (dist == "0K") or (dist == "Finish Net"):
            continue

        last_dist = last_dict[dist]
        last_mark = runner_dict[last_dist]
        curr_mark = runner_dict[dist]

        lk_array = get_likelihood(
            data=lk_data, last_dist=last_dist, last_mark=last_mark, curr_dist=dist,
            curr_mark=curr_mark, bin_mapping=np.ones(500), col_mapping=col_map, s2=s2
        )
        prior_array = bayes_dict["Prior"]
        bayes_dict[dist] = bayes_iter(prior=prior_array, lk=lk_array)
        bayes_dict["Prior"] = bayes_dict[dist]

    bayes_dict["Posterior"] = bayes_dict[person_marks[-2]]  # one before finish net
    bayes_dict.pop("Prior")
    return bayes_dict


def person_dict(
        person: pd.Series, marks: list, prior: np.array, lk_data, s2,
) -> Tuple[dict, int]:
    """Return both bayes table and actual value"""
    mk = _get_marks(marks_list=marks, zero_k=True, finish=False)
    assert person.shape[0] == len(mk)
    mk_idx = {dist: num for num, dist in enumerate(marks)}  # 0k + marks + finish
    actual = person[-1]
    runner_dict = {dist: person[mk_idx[dist]] for dist in mk}
    bayes_dict = full_bayes_dict(runner_dict=runner_dict, initial_prior=prior, lk_data=lk_data, s2=s2)
    return bayes_dict, actual


def initialize(
        train_file="processed_data/full_data_secs.csv",
        test_file="processed_data/full_data_secs.csv",
        max_fin: int = 500,
        store_prior: bool = True,
):
    """Initialize data, marks, and s2_mat"""
    train_df = pd.read_csv(train_file)
    train_data, train_info = get_train_set(train_df, zero_k=True)

    if store_prior:
        store_initial_prior(finish=train_data[:, -1] // 1, max_time=max_fin)
    s2_mat = get_s2_dict(train_data[:, -1], bin_mapping=np.ones(500), max_len=500)
    mks = _get_marks(marks_list=None, zero_k=True, finish=False)
    test_df = pd.read_csv(test_file)
    test_data, test_info = get_test_set(test_df, zero_k=True)
    return train_data, train_info, test_data, test_info, mks, s2_mat, max_fin


if __name__ == '__main__':
    train_data, train_info, test_data, test_info, marks, s2_matrix, max_finish = initialize()
    uninformed_prior = _prior_dist(informed=False, max_time=max_finish)
    print("start")
    for i, row in enumerate(test_data):
        informed_prior = _prior_dist(informed=True, max_time=max_finish)
        dict1 = person_dict(person=row, marks=marks, prior=informed_prior, lk_data=train_data, s2=s2_matrix)[0]
        print(i, test_info.iloc[i]["Name"])
        if i == 100:
            break
