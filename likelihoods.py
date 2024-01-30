
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import get_train_set, _get_intersection


def get_s2_dict(fin_data: np.array, bin_mapping: np.array, max_len: int = 500):
    """
    Return dictionary used to precompute part of likelihood calculation
    :param fin_data: array of finish times from train_data
    :param bin_mapping: mapping of each finish time to a bin size
    :return dict - {finish_time: [indexes of train_data where finish_time w/i bin size
    """
    mapping = {i: np.where(abs(fin_data - i) < bin_mapping[i])[0] for i in range(max_len)}
    return {k: set(v) for k, v in mapping.items() if v.shape[0] > 0}


def get_likelihood(
        data: np.array, last_dist: str, last_mark: float, curr_dist: str, curr_mark: float,
        bin_mapping: np.array, col_mapping: np.array, s2: np.array,
):
    """Return the likelihood array for a given `last_mark` and  `curr_mark`."""
    s1 = _subset_data(data=data, col_num=col_mapping[last_dist], mark=last_mark, diff=bin_mapping[round(last_mark)])
    s3 = _subset_data(data=data, col_num=col_mapping[curr_dist], mark=curr_mark, diff=bin_mapping[round(curr_mark)])
    lk_num, lk_den = np.zeros(500), np.zeros(500)

    for fin, indexes in s2.items():
        given = _get_intersection(s1, indexes)  # denominator
        if len(given) != 0:
            lk_num[fin] = len(_get_intersection(given, s3))
            lk_den[fin] = len(given)

    lks = np.divide(lk_num, lk_den, out=np.zeros(500), where=lk_den != 0)
    lks[np.isnan(lks)] = 0
    return lks


def _subset_data(data: np.array, col_num: int, mark: float, diff: int):
    """Subset dataframe, return set of all indexes close to mark"""
    return set(np.argwhere(abs(data[:, col_num] - mark) < diff).ravel())
    # return set(np.where(abs(data[:, col_num] - mark) < diff)[0])  # list of indexes satisfying


# def get_mapping(arr, max_len=500, sm=0.01, f=2, p=2):
#     x = np.array(range(max_len))
#     y1 = np.bincount(arr // 60, minlength=max_len)[:max_len]
#     y2 = y1 / y1.sum()
#     y = (y2 + sm) / (y2 + sm).sum()
#     return f * (x / (100_000 * y)) ** p


if __name__ == '__main__':
    df = pd.read_csv("processed_data/full_data_secs.csv")
    train_data, train_info = get_train_set(df, zero_k=True)
    m = ["0K", "5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K", "Finish Net"]
    col_map = {dist: num for num, dist in enumerate(m)}
    bin_map = np.ones(500)
    s2_matrix = get_s2_dict(train_data[:, -1], bin_mapping=np.ones(500), max_len=500)
    lk = get_likelihood(data=train_data, last_dist="5K", last_mark=24, curr_dist="10K",
                        curr_mark=48, bin_mapping=bin_map, col_mapping=col_map, s2=s2_matrix)
    plt.plot(lk)
    plt.show()
    print('done')