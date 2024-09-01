
# utils.py: utility functions

import numpy as np
import random
import arviz as az
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous, norm
from bayes_models.model1 import LinearModel as LinearModel1
from bayes_models.model2 import LinearModel as LinearModel2
from bayes_models.model3 import LinearModel as LinearModel3


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


#######
# was in utilsb.py

marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]

last_map = {
    "10K": "5K", "15K": "10K", "20K": "15K",
    "25K": "20K", "30K": "25K", "35K": "30K",
    "40K": "35K", "Finish Net": 42_195,
}

conv1 = {
    "5K": 5_000, "10K": 10_000, "15K": 15_000, 
    "20K": 20_000, "25K": 25_000, "30K": 30_000,
    "35K": 35_000, "40K": 40_000, "Finish Net": 42_195,
}

def time_to_pace(time, dist):
    secs = time * 60
    return conv1[dist] / secs


def pace_to_time(pace, dist):
    secs = conv1[dist] / pace
    return secs / 60

def process_df(data):
    new_idx, new_dist, new_mark, new_fin, new_last = [], [], [], [], []
    new_age, new_gender, new_year = [], [], []
    for dist in marks:
        new_idx.extend(list(data.index))
        new_dist.extend([dist] * len(data))
        new_mark.extend(conv1[dist] / data[dist])
        new_fin.extend(conv1["Finish Net"] / data["Finish Net"])
        if dist == "5K":
            last = data["5K"]
        else:
            last = data[dist] - data[last_map[dist]]
        new_last.extend(conv1["5K"] / last)
        
        new_age.extend(data["Age"])
        new_gender.extend(data["M/F"])
        new_year.extend(data["Year"])

    return pd.DataFrame({
        "id": new_idx, "dist": new_dist, "curr_pace": new_last, "total_pace": new_mark, "finish": new_fin,
        "age": new_age, "gender": new_gender, "year": new_year, 
    })

def get_data(filepath="full_data_secs.csv", size_train=50, size_test=50, train_tup=(2022, 2023), test_tup=(2023, 2024)):
    d = pd.read_csv(filepath)
    xtrain = process_df(d[d["Year"].isin(range(*train_tup))])
    xtest = process_df(d[d["Year"].isin(range(*test_tup))])

    train_ids = random.sample(set(xtrain["id"]), size_train)
    xtrain = xtrain[xtrain["id"].isin(train_ids)]

    test_ids = random.sample(set(xtest["id"]), size_test)
    xtest = xtest[xtest["id"].isin(test_ids)]

    # xtrain = xtrain.groupby('dist', group_keys=False).apply(lambda x: x.sample(size_train))
    # xtest = xtest.groupby('dist', group_keys=False).apply(lambda x: x.sample(size_test))
    # nucr_test = process_df(pd.read_csv("nucr_runners.csv"))
    return xtrain, xtest,#nucr_test

#######

def get_models_and_traces(
    # LinearModel1,
    # LinearModel2,
    model1 = "traces/linear_model1.nc",
    trace1 = "traces/linear_trace1.nc",
    model2 = "traces/linear_model2.nc",
    trace2 = "traces/linear_trace2.nc",
    model3: bool = False,

):
    models = [LinearModel1.load(model1), LinearModel2.load(model2)]
    traces = [az.from_netcdf(trace1), az.from_netcdf(trace2)]

    if model3:
        models.append(LinearModel1.load("traces/linear_model3.nc"))
        traces.append(az.from_netcdf("traces/linear_trace3.nc"))
    
    return models, traces

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
