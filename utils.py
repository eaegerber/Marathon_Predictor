
# utils.py: utility functions

import json
import numpy as np
import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, norm, lognorm  # , exponnorm, invgauss, norminvgauss


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


def store_initial_prior(
        data: pd.DataFrame,
        max_time: int = 500,
        path: str = "processed_data/informed_prior.csv"
):
    """Store the initial prior distribution from data in a file"""
    prior = np.array(lognorm_likelihood(dict(data["Finish Net"].value_counts())))
    # prior = (data["Finish Net"].value_counts() / len(data["Finish Net"])).rename("Prior")
    # prior = np.array(pd.DataFrame(prior, index=range(max_time + 1)).fillna(0))
    prior.tofile(path, sep=',')
    return prior


def get_training_set(people: pd.DataFrame):
    """Return people from before this year"""
    return people[people["Year"] != 2023]


def get_test_set(people: pd.DataFrame):
    """Return people from this year"""
    return people[people["Year"] == 2023]


def store_lk_json(dists: list, data: pd.DataFrame):
    """Store likelihood counts into a single json {dist : {mark : {finish : count} } }"""
    all_json = {}
    for dist in dists:
        dist_json = {}
        counts_data = data.groupby([dist, "Finish Net"])["Name"].count().reset_index().values
        for mark, fin, count in counts_data:
            mark, fin, count = int(mark), int(fin), int(count)
            if fin not in dist_json.keys():
                dist_json[fin] = {}
            dist_json[fin][mark] = count

        all_json[dist] = dist_json

    with open("likelihood_tables/all_likelihoods.json", "w") as file:
        json.dump(all_json, file)

    return


def store_processed_likelihoods(max_length: int = 500):
    """Store likelihoods fit to a lognorm distribution"""
    with open("likelihood_tables/all_likelihoods.json") as file:
        dct = json.load(file)

    new_dict = {}
    for dist in dct.keys():
        lk_table = np.zeros(shape=(max_length, max_length), dtype=int)
        for finish in dct[dist].keys():
            for mark, count in dct[dist][finish].items():
                if int(finish) < max_length:
                    lk_table[int(finish)][int(mark)] = int(count)

        fin_counts = lk_table.sum(axis=1)
        lk_tbl = (lk_table.T / fin_counts).T
        lk_tbl2 = pd.DataFrame(lk_tbl).fillna(0)

        new_dict[dist] = {i: list(lk_tbl2[i]) for i in range(max_length)}  # new_lognorm_lk

    for dist, dct in new_dict.items():
        filename = f"likelihood_tables/likelihoods_{dist}.json"
        with open(filename, "w") as file:
            json.dump(dct, file)
        print('loaded: ', dist)

    return


def lognorm_likelihood(dct: dict, max_length: int = 500) -> list:
    """Compute lognorm likelihood from a dict of counts for each finish.
    Return the numpy array of finish time probabilities"""
    finish_list = np.concatenate([[int(k)] * v for k, v in dct.items()])  # flatten values (lists) into single list
    shape, loc, scale = lognorm.fit(finish_list, scale=np.exp(np.log(finish_list).mean()))
    m = lognorm(shape, loc, scale)
    val_counts = m.pdf(range(max_length))
    return val_counts.tolist()


def read_in_jsons(marks: list) -> dict:
    """Read in dict from file for each mark"""
    new_dict = {}
    for dist in marks:
        with open(f"likelihood_tables/likelihoods_{dist}.json") as file:
            new_dict[dist] = json.load(file)

    return new_dict


if __name__ == '__main__':
    df = pd.read_csv("processed_data/full_data_mins.csv")
    data_series = df["Finish Net"]

    plot_dist(data=df, checkpoint="Finish Net", ticks=(120, 240, 423), save="plot_dist2")
    # plot_dist(data=df, checkpoint="5K", ticks=(30, 60, 120), save="plot_dist")
    fit_model(data_series, model=norm, plot_range=(data_series.min(), data_series.max()))
    plt.show()
