
# utils.py: utility functions
from typing import Union
import numpy as np
import pandas as pd
import seaborn as sns
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


def int_to_str_time(time: int, no_secs: bool = False)  -> Union[str, None]:
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

        if no_secs:
            return hrs + ":" + mins
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
        plt.savefig(f"analysis/{save}.png")

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

def get_data(racename="bos", size_train=50, size_test=50, train_lis=[2022], test_lis=[2023], save=False):
    """Get and process data from filepath. After processing data, this function samples the training and test data
    based on training and test data specifications for size and years. Returns train and test data"""
    d = pd.read_csv(f"processed_data/full_data_{racename}.csv")
    train_years, test_years =  d[d["Year"].isin(train_lis)], d[d["Year"].isin(test_lis)]

    if size_train != None:
        train_years = train_years.sample(n=size_train, random_state=2025, replace=False)
    
    if size_test != None:
        test_years = test_years.sample(n=size_test, random_state=2025, replace=False)
    
    xtrain = process_df(train_years)
    xtest = process_df(test_years)

    if save:
        xtrain.to_csv(f"processed_data/train_{racename}.csv")
        xtest.to_csv(f"processed_data/test_{racename}.csv")
    return xtrain, xtest

#######

def get_preds(test_data, stan_data, feats_lis, beta_lis, name="stan_pred", propleft=False, full=False):
    """Get predictions from test data using stan results. The feat_lis columns in test_data correspond
    with the beta_lis columns in stan_data."""
    test_new = test_data.copy()
    d1 = test_new[feats_lis].copy()
    d2 = stan_data[beta_lis].T.copy()

    norm_mean = stan_data["alpha"] + d1.dot(d2.values)

    if full:
        if propleft: 
            norm_std = np.outer(test_new["propleft"], stan_data["sigma"])
        else:
            norm_std = stan_data["sigma"]
        preds = np.random.normal(norm_mean, norm_std)
        return preds
    else:
        preds = norm_mean.mean(axis=1) #preds.mean(axis=1)
        return preds
    
def other_stats(data, finish):
    """Return overall RMSE and R-squared values for specified columns in data"""
    ftime = (42195/60) / finish
    tss = (((ftime) - (ftime).mean()) ** 2).sum()
    return data.apply(lambda x: ((x ** 2).mean() ** 0.5, 1 - ((x ** 2).sum()/ tss)))  # overall rmse, rsquared

def get_table(test_data, model_preds):
    """Get table that outpus all information to compare models"""
    test_new = test_data.copy()
    y_true = (42195 / 60) / test_new["finish"]
    extrap = (42195 / 60) / test_new["total_pace"]
    
    test_new["extrap"] = extrap - y_true
    for name, pred in model_preds.items():
        test_new[name] = ((42195 / 60) / pred) - y_true

    return test_new

def other_plots(test_data, m_preds):
    marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    for mark in marks:
        sns.kdeplot(test_data[test_data["dist"] == mark]["rstan2d"], label=mark)
    plt.xlim(-20, 20)
    plt.legend()
    plt.show()

    for m in m_preds:
        sns.kdeplot(test_data[m], label=m)

    plt.xlim(-20, 20)
    plt.legend()

def plot_rmse(test_data: pd.DataFrame, labels: list, save_name: str = "bos"):
    """Create table and plot to compare the RMSE for multiple mdoels. Labels specifies the
    models to be shown."""
    colors = [f"C{i}" for i in range(len(labels))]
    styles = '.-'
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    table_group = test_data.groupby(["dist"])[labels].apply(lambda x: (x ** 2).mean() ** 0.5).loc[mks]
    table_group.plot(label=table_group.columns, style=styles, linewidth=2, grid=True, alpha=0.8, color=colors)

    plt.xlabel("Distance Into Race (km)")
    plt.ylabel("Prediction Error (RMSE), in minutes")
    plt.xticks(rotation=60)
    plt.title("Average Error For Each Model")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"analysis/{save_name}_rmse.png", bbox_inches="tight")
    print(f"File saved: analysis/{save_name}_rmse.png")
    plt.close()
    return table_group

def add_intervals_to_test(data_tbl, m_preds, pred_names):
    data = data_tbl.copy()
    for pred_name in pred_names:
        for conf, lower, upper in [(50, 25, 75), (80, 10, 90), (95, 2.5, 97.5)]:
            test_true = (42195 / 60) / data["finish"]
            b1, b2 = np.percentile(m_preds[pred_name], [lower, upper], axis=1)
            data[f"{pred_name}-lower{conf}"] = b1
            data[f"{pred_name}-upper{conf}"] = b2
            data[f"{pred_name}-size{conf}"] = b2 - b1
            data[f"{pred_name}-in{conf}"] = (test_true < b2) & (test_true > b1)
    return data


def plot_interval_check(itbl: pd.DataFrame, pred_names: list, intervals: list = [50, 80, 95], 
                        linestyles = ["-.", "--.", ":."], save_name: str = "bos"):
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
    plt.savefig(f"analysis/{save_name}_interval_check.png", bbox_inches="tight")
    print(f"File saved: analysis/{save_name}_interval_check.png")
    plt.close()
    return big_table


def plot_interval_sizes(itbl: pd.DataFrame, pred_names: list, intervals: list = [50, 80, 95], 
                        linestyles = ["-.", "--.", ":."], save_name: str = "bos"):
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

    plt.savefig(f"analysis/{save_name}_interval_sizes.png", bbox_inches="tight")
    print(f"File saved: analysis/{save_name}_interval_sizes.png")
    plt.close()
    return big_table

if __name__ == '__main__':
    df = pd.read_csv("processed_data/full_data_bos.csv")
    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    df[marks + ["Finish Net"]] = ((df[marks + ["Finish Net"]] // 60) + 1).astype(int)
    data_series = df["Finish Net"]
    # plot_dist(data=df, checkpoint="Finish Net", ticks=(120, 240, 423), save="plot_dist2")
    plot_dist(data=df, checkpoint="Finish Net", ticks=[60 * x for x in range(10)], save="plot_dist")
    # plot_dist(data=df, checkpoint="5K", ticks=(30, 60, 120), save="plot_dist")
    plt.show()
