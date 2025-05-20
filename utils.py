
# utils.py: utility functions
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

np.random.seed(2024)

def str_to_int_time(time: str)  -> Union[int, None]:
    """Convert string time (HH:MM:SS) to int time"""
    try:
        times = time.split(":")[::-1]
        seconds = sum(x * int(time) for x, time in zip([1, 60, 3600], times))
        return int(seconds)
    except ValueError:
        return None
    except AttributeError:
        return None


def int_to_str_time(time: int)  -> Union[str, None]:
    """Convert int time (in mins) to str time (MM:SS) or (HH:MM:SS)"""
    secs = int(time % 60)
    time2 = int((time - secs) / 60)  # should be int
    if time2 < 60:
        mins, secs = str(time2).zfill(2), str(secs).zfill(2)
        return "00" + ":" + mins + ":" + secs
        # return mins + ":" + secs
    else:
        mins = int((time2 % 60))
        hrs = int((time2 - mins) / 60)
        hrs, mins, secs = str(hrs).zfill(2), str(mins).zfill(2), str(secs).zfill(2)
        return hrs + ":" + mins + ":" + secs


def binning(data: pd.Series):
    """Return number of bins for series of data (for each integer)"""
    return int(data.max()) - int(data.min()) + 1


def plot_dist(
        data: pd.DataFrame,
        checkpoint: str = "Finish Net",
        ticks: tuple = (60, 120, 180, 240, 300, 360, 420),
        save = None #: Union[str, None] = None
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
    """Convert time to pace"""
    secs = time * 60
    return conv1[dist] / secs


def pace_to_time(pace, dist):
    """Convert pace to time"""
    secs = conv1[dist] / pace
    return secs / 60

def process_df(data):
    """Processes data. Input dataframe has columns=[Age, M/F, 5K, 10K, ... 40K, HALF, Finish Net, Year], while
    output dataframe has columns=[dist, curr_pace, total_pace, finish, age, gender, year]"""
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

    new_df = pd.DataFrame({
        "id": new_idx, "dist": new_dist, "curr_pace": new_last, "total_pace": new_mark, "finish": new_fin,
        "age": new_age, "gender": new_gender, "year": new_year
    })
    new_df["prop"] = new_df["dist"].apply(lambda x: conv1[x] / conv1["Finish Net"])
    new_df["propleft"] = 1 - new_df["prop"]
    new_df["male"] = (new_df['gender'] == "M").astype(int)
    new_df["propxcurr"] = new_df["prop"] * new_df["curr_pace"]
    new_df["malexage"] = new_df["male"] * new_df["age"]
    return new_df

def get_data(filepath="full_data_secs.csv", size_train=50, size_test=50, train_tup=(2022, 2023), test_tup=(2023, 2024)):
    """Get and process data from filepath. After processing data, this function samples the training and test data
    based on training and test data specifications for size and years. Returns train and test data"""
    d = pd.read_csv(filepath)
    xtrain = process_df(d[d["Year"].isin(range(*train_tup))])
    xtest = process_df(d[d["Year"].isin(range(*test_tup))])

    train_ids = np.random.choice(np.array(list(set(xtrain["id"]))), size_train, replace=False)
    xtrain = xtrain[xtrain["id"].isin(train_ids)]

    test_ids = np.random.choice(np.array(list(set(xtest["id"]))), size_test, replace=False)
    xtest = xtest[xtest["id"].isin(test_ids)]

    # xtrain = xtrain.groupby('dist', group_keys=False).apply(lambda x: x.sample(size_train))
    # xtest = xtest.groupby('dist', group_keys=False).apply(lambda x: x.sample(size_test))
    return xtrain, xtest

#######

def get_preds(test_data, stan_data, feats_lis, beta_lis, name="stan_pred", propleft=False, full=False):
    """Get predictions from test data using stan results. The feat_lis columns in test_data correspond
    with the beta_lis columns in stan_data."""
    test_new = test_data.copy()
    d1 = test_new[feats_lis].copy()
    d2 = stan_data[beta_lis].T.copy()

    norm_mean = stan_data["alpha"] + d1.dot(d2.values)
    if propleft: 
        norm_std = np.outer(test_new["propleft"], stan_data["sigma"])
    else:
        norm_std = stan_data["sigma"]
    preds = np.random.normal(norm_mean, norm_std)
    
    if full:
        return preds
    else:
        preds = preds.mean(axis=1)
        test_new[name] = preds
        return test_new 
    

def get_table(test_data, old="stan_pred", new="stan"):
    """Get table that outpus all information to compare models"""
    y_true = (42195 / 60) / test_data["finish"]
    preds = (42195 / 60) / test_data[old]
    extrap = (42195 / 60) / test_data["total_pace"]
    test_data[new] = preds - y_true
    test_data["extrap"] = extrap - y_true
    return test_data

def plot_rsme(test_data: pd.DataFrame, labels: list, save_name: str = "all_errors"):
    """Create table and plot to compare the RMSE for multiple mdoels. Labels specifies the
    models to be shown."""
    colors = [f"C{i}" for i in range(len(labels))]
    styles = '--'
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    table_group = test_data.groupby(["dist"])[labels].apply(lambda x: (x ** 2).mean() ** 0.5).loc[mks]
    table_group.plot(label=table_group.columns,  style=styles, linewidth=3, grid=True, alpha=0.75, color=colors)

    plt.xlabel("Distance Into Race")
    plt.ylabel("Prediction Error (RMSE)")
    plt.xticks(rotation=60)
    plt.title("Average Error For Each Model")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"analysis/{save_name}.png", bbox_inches="tight")
    return table_group


def plot_interval_check(itbl: pd.DataFrame, pred_names: list, intervals: list = [50, 80, 95], 
                        linestyles = ["-.", "--.", ":."]):
    """Plot the interval check for each model specified in pred_names"""
    n = len(pred_names)
    colors = [f"C{i}" for i in range(len(pred_names))] * len(intervals)
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    tables = []
    for conf in intervals:
        sublabels = [f"{p}-in{conf}" for p in pred_names]
        result_tbl = itbl.groupby(["dist"])[sublabels].apply(lambda x: x.sum() / len(x)).loc[mks]
        tables.append(result_tbl)

    plt.figure(figsize=(40, 40))
    big_table = pd.concat(tables, axis=1)
    
    styles = []
    for s in linestyles:
        styles += [s] * n
    # styles = ["-."] * n + ["--."] * n + [":."] * n
    ax = big_table.plot(label=big_table.columns,  style=styles , linewidth=2, grid=True, alpha=0.75, color=colors)

    plt.xlabel("Distance Into Race")
    plt.ylabel("Proportion of Actual Finish Times Within Credible Interval")
    plt.xticks(rotation=60)
    plt.ylim(0.2, 1)
    plt.title("Proportion Within Interval Over Different Distances")
    plt.grid(True)
    plt.legend()
    leg1 = plt.legend([Line2D([0,1],[0,1],linestyle=s, color='black') for s in ["-.", "--", ":"]], ["50", "80", "95"], loc="lower center")
    ax.add_artist(leg1)
    leg2 = plt.legend([Line2D([0,1],[0,1], color=c) for c in colors[:n]], pred_names, loc="lower right")
    ax.add_artist(leg2)
    plt.savefig("analysis/interval_check.png", bbox_inches="tight")
    return big_table


def plot_interval_sizes(itbl: pd.DataFrame, pred_names: list, intervals: list = [50, 80, 95], 
                        linestyles = ["-.", "--.", ":."]):
    """Plot the interval sizes for each model specified in pred_names"""
    n = len(pred_names)
    colors = [f"C{i}" for i in range(len(pred_names))] * len(intervals)
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    tables = []
    for conf in intervals:
        sublabels = [f"{p}-size{conf}" for p in pred_names]
        result_tbl = itbl.groupby(["dist"])[sublabels].apply(lambda x: x.sum() / len(x)).loc[mks]
        tables.append(result_tbl)

    plt.figure(figsize=(30, 20))
    big_table = pd.concat(tables, axis=1)
    
    styles = []
    for s in linestyles:
        styles += [s] * n
    ax = big_table.plot(label=big_table.columns,  style=styles , linewidth=2, grid=True, alpha=0.75, color=colors)

    plt.xlabel("Distance Into Race")
    plt.ylabel("Credible Interval Sizes")
    plt.xticks(rotation=60)
    plt.title("Credible Interval Sizes")
    plt.grid(True)

    leg1 = plt.legend([Line2D([0,1],[0,1],linestyle=s, color='black') for s in ["-.", "--", ":"]], ["50", "80", "95"], loc="upper center")
    ax.add_artist(leg1)
    leg2 = plt.legend([Line2D([0,1],[0,1], color=c) for c in colors[:n]], pred_names, loc="upper right")
    ax.add_artist(leg2)

    plt.savefig("analysis/interval_sizes.png", bbox_inches="tight")
    return big_table

if __name__ == '__main__':
    df = pd.read_csv("processed_data/full_data_secs.csv")
    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    df[marks + ["Finish Net"]] = ((df[marks + ["Finish Net"]] // 60) + 1).astype(int)
    data_series = df["Finish Net"]
    # plot_dist(data=df, checkpoint="Finish Net", ticks=(120, 240, 423), save="plot_dist2")
    plot_dist(data=df, checkpoint="Finish Net", ticks=[60 * x for x in range(10)], save="plot_dist")
    # plot_dist(data=df, checkpoint="5K", ticks=(30, 60, 120), save="plot_dist")
    plt.show()
