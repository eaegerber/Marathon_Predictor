import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_data, binning, int_to_str_time#, Union
random.seed(2024)

def get_plot_dist(races=["bos", "nyc", "chi"], yrs=[2022, 2023]):
    ticks = (60, 120, 180, 240, 300, 360, 420, 480, 540, 600)
    train_list = [(r, pd.read_csv(f"processed_data/full_data_{r}.csv")) for r in races]

    for lbl, train_data in train_list:
        # minutes_dist = train_data[train_data["Year"].isin(yrs)]["Finish Net"] // 60
        minutes_dist = train_data["Finish Net"] // 60
        bins = binning(minutes_dist)
        minutes_dist.hist(bins=bins, alpha=0.6, density=True, label=lbl)
    labels = [int_to_str_time(60 * t, no_secs=True) for t in ticks]
    plt.xticks(ticks, labels=labels)
    plt.xlim(ticks[1] - 30, ticks[-2] + 30)
    plt.xlabel(f"Time (HH:MM)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Marathon Finish Times")
    plt.legend()
    plt.savefig(f"analysis/plot_dist.png")
    plt.close()


def get_data_scatter():
    train2, _ = get_data(filepath="processed_data/full_data_bos.csv", size_train=6064, train_tup=(0, 2026), size_test=400)
    table1 = train2[train2["dist"].isin(["10K", "20K", "30K"])]
    table1[["total_pace", "finish"]] = table1[["total_pace", "finish"]].apply(lambda x: (42195 / 60) / x)

    train_ids = np.random.choice(np.array(list(set(table1["id"]))), 1000, replace=False)
    table2 = table1[table1["id"].isin(train_ids)]

    sns.jointplot(data=table2, x="total_pace", y="finish", hue="dist")
    plt.plot([100, 450], [100, 450], color="red", label='trad est', alpha=0.4)
    plt.xlabel("Finish Estimate Extrapolated from Total Pace So Far (HH:MM)")
    plt.ylabel("True Finish Time (HH:MM)")

    ticks = (100, 150, 200, 250, 300, 350, 400, 450)
    labels = [int_to_str_time(60 * t, no_secs=True) for t in ticks]
    plt.xticks(ticks, labels=labels)
    plt.yticks(ticks, labels=labels)

    for dist, color in [("10K", "blue"), ("20K", "orange"), ("30K", "green")]:
        small_data = table2[table2["dist"] == dist]#[["total_pace", "finish"]]
        m, b = np.polyfit(small_data["total_pace"], small_data["finish"], 1)
        x = np.array([100, 450])
        plt.plot(x, m*x + b, alpha=0.4, label=f"{dist} fit")

    plt.ylim(90, 460)
    plt.xlim(90, 460)
    # plt.title("Comparing Extrapolated Estimates to True Finish At Different Stages of Race")
    plt.grid()
    plt.legend()
    # plt.savefig("analysis/data_scatter_.png", bbox_inches="tight")

if __name__ == "__main__":
    print('start')
    get_plot_dist()
    # get_data_scatter()
    print('fin')