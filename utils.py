
# utils.py: utility functions

import numpy as np
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, norm


def str_to_int_time(time: str) -> Union[int, None]:
    """Convert string time (HH:MM:SS) to int time"""
    try:
        times = time.split(":")[::-1]
        seconds = sum(x * int(time) for x, time in zip([1, 60, 3600], times))
        return int(seconds)
    except ValueError:
        return None
    except AttributeError:
        return None


def int_to_str_time(time: int) -> Union[str, None]:
    """Convert int time (in mins) to str time"""
    mins = int(time % 60)
    hrs = int((time - mins) / 60)  # should be int
    hrs, mins = str(hrs).zfill(2), str(mins).zfill(2)
    return hrs + ":" + mins


def binning(data: pd.Series):
    """Return number of bins for series of data (for each integer)"""
    return int(data.max()) - int(data.min()) + 1


def plot_dist(
        data: pd.DataFrame,
        checkpoint: str = "Finish Net",
        ticks: tuple = (60, 120, 180, 240, 300, 360, 420),
        save: Union[str, None] = None
) -> None:
    """Plot the distribution for a given checkpoint"""
    plt.figure(figsize=(10, 8))
    minutes_dist = data[checkpoint]  # checkpoint time in mins, rounded up

    bins = binning(minutes_dist)
    minutes_dist.hist(bins=bins)
    labels = [int_to_str_time(t) for t in ticks]
    plt.xticks(ticks, labels=labels)
    plt.xlabel(f"Time")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Marathon {checkpoint} Times")
    if isinstance(save, str):
        plt.savefig(f"plots/{save}.png")


def fit_model(data: pd.Series, model: rv_continuous, plot_range: tuple = (120, 300)):
    """Fit a distribution to a specified model type"""
    plt.figure(figsize=(10, 8))

    bins = binning(data)
    params = model.fit(data)
    print(f"{params} -> {model}")
    d = model(*params)

    data.hist(bins=bins, density=True)
    x = np.linspace(*plot_range, 1000)
    plt.plot(x, d.pdf(x), color="orange")


def get_train_set(people: pd.DataFrame, zero_k: bool = False):
    """Return people from before this year"""
    if zero_k:
        people["0K"] = 0

    marks_ls = _get_marks(marks_list=None, zero_k=zero_k, finish=True)
    people = people[people["Year"] != 2023]
    people_info = people[["Name", "Age", "M/F"]]
    people_array = (np.array(people[marks_ls]) / 60).round(2)
    return people_array, people_info


def get_test_set(people: pd.DataFrame, zero_k: bool = False):
    """Return people from this year"""
    if zero_k:
        people["0K"] = 0

    marks_ls = _get_marks(marks_list=None, zero_k=zero_k, finish=True)
    people = people[people["Year"] == 2023]
    people_info = people[["Name", "Age", "M/F"]]
    people_array = (np.array(people[marks_ls]) / 60).round(2)
    return people_array, people_info


def store_initial_prior(finish: np.array, max_time: int = 500, path: str = "processed_data/informed_prior.csv"):
    """Store the initial prior distribution from data in a file"""
    # counts = dict((data["Finish Net"] // 60).value_counts())
    counts = dict(enumerate(np.bincount(finish.astype(int))))
    counts = [counts[idx] if idx in counts.keys() else 0 for idx in range(max_time)]
    prior = np.array(counts) / sum(counts)
    prior.tofile(path, sep=',')
    return prior


def _prior_dist(informed: bool = True, max_time: int = 500) -> np.array:
    """Returns the prior distribution for the runner as a numpy array"""
    if informed:
        prior = np.loadtxt("processed_data/informed_prior.csv", delimiter=',')
    else:
        prior = np.ones(max_time) / max_time

    return prior


def _get_marks(marks_list: Union[list, None], zero_k: bool = False, finish: bool = False):
    """Order and return the subset of marks specified in marks_list"""
    all_marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    if not marks_list:
        marks_list = all_marks
    marks_ls = [m for m in all_marks if m in marks_list]
    if zero_k and marks_ls[0] != "0K":
        marks_ls = ["0K"] + marks_list  # TODO fix
    if finish:
        marks_ls = marks_ls + ["Finish Net"]
    return marks_ls


def _get_intersection(a1, a2) -> np.array:
    if len(a1) == 286777:
        return a2
    if len(a2) == 286777:
        return a1
    if len(a1) < len(a2):
        return a1.intersection(a2)
    else:
        return a1.intersection(a2)


if __name__ == '__main__':
    df = pd.read_csv("processed_data/full_data_secs.csv")
    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    df[marks + ["Finish Net"]] = ((df[marks + ["Finish Net"]] // 60) + 1).astype(int)
    data_series = df["Finish Net"]

    plot_dist(data=df, checkpoint="Finish Net", ticks=(120, 240, 423), save="plot_dist2")
    plot_dist(data=df, checkpoint="Finish Net", ticks=[60 * x for x in range(10)], save="plot_dist")
    # plot_dist(data=df, checkpoint="5K", ticks=(30, 60, 120), save="plot_dist")
    fit_model(data_series, model=norm, plot_range=(data_series.min(), data_series.max()))
    plt.show()
