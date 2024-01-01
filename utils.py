
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


def get_training_set(people: pd.DataFrame):
    """Return people from before this year"""
    return people[people["Year"] != 2023]


def get_test_set(people: pd.DataFrame):
    """Return people from this year"""
    return people[people["Year"] == 2023]


def store_initial_prior(data: pd.DataFrame, max_time: int = 500, path: str = "processed_data/informed_prior.csv"):
    """Store the initial prior distribution from data in a file"""
    counts = {int(2 * k): v for k, v in (((data["Finish Net"] + 15) // 30) / 2).value_counts().to_dict().items()}
    # counts = dict(data["Finish Net"].value_counts())
    counts = [counts[idx] if idx in counts.keys() else 0 for idx in range(max_time)]
    prior = np.array(counts) / sum(counts)
    prior.tofile(path, sep=',')
    return prior


def round_df(secs_df: pd.DataFrame, marks_list: list):
    secs_df = secs_df.copy()
    secs_df[marks_list] = (((secs_df[marks_list] + 15) // 30) / 2)
    secs_df["Finish Net"] = secs_df["Finish Net"] // 60
    return secs_df


if __name__ == '__main__':
    df = pd.read_csv("processed_data/full_data_secs.csv")
    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    df[marks + ["Finish Net"]] = ((df[marks + ["Finish Net"]] // 60) + 1).astype(int)
    data_series = df["Finish Net"]

    plot_dist(data=df, checkpoint="Finish Net", ticks=(120, 240, 423), save="plot_dist2")
    # plot_dist(data=df, checkpoint="5K", ticks=(30, 60, 120), save="plot_dist")
    fit_model(data_series, model=norm, plot_range=(data_series.min(), data_series.max()))
    plt.show()
