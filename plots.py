
# plots.py: functionality to generate miscellaneous plots

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_data, binning, int_to_str_time#, Union
random.seed(2024)

def get_plot_dist(races=["bos", "nyc", "chi",], yrs=[2022, 2023]):
    """Plot distribution of finish times for a list of races in specified years"""
    ticks = (60, 120, 180, 240, 300, 360, 420, 480, 540, 600)
    train_list = [(r, pd.read_csv(f"processed_data/full_data_{r}.csv")) for r in races]
    cmap = {"bos": "darkgreen", "nyc": "blue", "chi": "crimson"}

    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(10)
    for lbl, train_data in train_list:
        # minutes_dist = train_data[train_data["Year"].isin(yrs)]["Finish Net"] // 60
        minutes_dist = train_data["Finish Net"] // 60
        bins = binning(minutes_dist)
        minutes_dist.hist(bins=bins, alpha=0.2, density=True, color=cmap[lbl], ax=ax)
        d2 = np.bincount(minutes_dist)
        ax.plot(range(len(d2)), d2 / sum(d2), color=cmap[lbl], linewidth=0.8, label=lbl)
    
    ax.set_facecolor(("orange", 0.05))
    labels = [int_to_str_time(60 * t, no_secs=True) for t in ticks]
    
    plt.xticks(ticks, labels=labels)
    plt.xlim(ticks[1] - 15, ticks[-2] - 45)
    plt.xlabel(f"Time (HH:MM)")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Marathon Finish Times")
    plt.legend()
    plt.savefig(f"analysis/plot_dist.png")
    plt.close()


def get_extrap_scatter(racename="bos", sample_size=1000, yrs=[2022, 2023], mks=["10K", "20K", "30K"]):
    """Get scatter plot comparing extrapolated finish times to actual finish times for a given race"""
    train_data, _ = get_data(racename=racename, size_train=sample_size, train_lis=yrs)
    table = train_data[train_data["dist"].isin(mks)].copy()
    table[["total_pace", "finish"]] = table[["total_pace", "finish"]].apply(lambda x: (42195 / 60) / x)
    
    if racename == "bos":
        ticks = (90, 150, 210, 270, 330, 390)
    else:
        ticks = (90, 150, 210, 270, 330, 390, 450, 510)

    sns.set_palette("mako", n_colors=len(mks)) #crest #mako
    print(sns.color_palette())
    sns.jointplot(data=table, x="total_pace", y="finish", hue="dist", alpha=0.8)# hueorder=["red", "blue", "green"])
    plt.plot([ticks[0] - 30, ticks[-1] + 30], [ticks[0] - 30, ticks[-1] + 30], color="red", label='trad est', alpha=0.6)
    plt.xlabel("Finish Estimate Extrapolated from Total Pace So Far (HH:MM)")
    plt.ylabel("True Finish Time (HH:MM)")

    labels = [int_to_str_time(60 * t, no_secs=True) for t in ticks]
    plt.xticks(ticks, labels=labels, rotation=0)
    plt.yticks(ticks, labels=labels, rotation=0)

    for i, dist in enumerate(mks):
        small_data = table[table["dist"] == dist]
        m, b = np.polyfit(small_data["total_pace"], small_data["finish"], 1)
        x = np.array([ticks[0] - 30, ticks[-1] + 30])
        plt.plot(x, m*x + b, alpha=0.8, label=f"{dist} fit", linestyle="--", color=sns.color_palette()[i])

    plt.xlim(ticks[0] - 30, ticks[-1] + 30)
    plt.ylim(ticks[0] - 30, ticks[-1] + 30)
    # plt.title("Comparing Extrapolated Estimates to True Finish At Different Stages of Race")
    plt.grid()
    plt.legend()#ncols=2
    plt.savefig(f"analysis/{racename}_data_scatter.png", bbox_inches="tight")

if __name__ == "__main__":
    print('start')
    get_plot_dist()
    get_extrap_scatter("bos") # , mks=["5K", "15K", "25K", "35K"])
    get_extrap_scatter("nyc")
    get_extrap_scatter("chi")
    print('fin')