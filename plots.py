
# plots.py: functionality to generate miscellaneous plots

import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from utils import get_data, binning, int_to_str_time, rmse_table, group_data #, Union
random.seed(2024)

def get_plot_dist(races=["bos", "nyc", "chi",], yrs=[2021, 2022, 2023, 2024]):
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
    
    # ax.set_facecolor(("orange", 0.05))
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


def plot_rmse(test_data: pd.DataFrame, labels: list, save_name: str = "bos", bar=True, rnd=3):
    """Create table and plot (line or bar) to compare the RMSE for multiple mdoels. Labels
      specifies the models to be shown."""
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(10)

    colors = [f"C{i}" for i in range(len(labels))]
    # sns.set_palette("viridis", n_colors=len(colors), desat=0.8)
    table_group = rmse_table(test_data, labels).sort_index(axis=1)
    if bar:
        table_group.plot(width=0.6, alpha=0.8, edgecolor="black", linewidth=0.3, ax=ax, legend=False, kind="bar") #, color=colors)
        suff = "_bar"
    else:
        table_group.plot(label=table_group.columns, style='.-', linewidth=2, grid=True, alpha=0.8, color=colors, ax=ax)
        suff = "_line"
        
    # fig.patch.set_facecolor(('yellow', 0.05)) # This changes the grey to white
    #ax.set_facecolor(("orange", 0.05))
    plt.xlabel("Distance Into Race (km)")
    plt.ylabel("Prediction Error (RMSE), in minutes")
    plt.xticks(rotation=60)
    plt.title("Average Error For Each Model")
    plt.grid(True)
    plt.legend()
    if save_name != "":
        plt.savefig(f"analysis/{save_name}_rmse{suff}.png", bbox_inches="tight")
        print(f"File saved: analysis/{save_name}_rmse{suff}.png")
    plt.close()

    for lbl in labels:
        if lbl != "extrap":
            table_group[f"pcnt_{lbl}"] = 1 - (table_group[lbl] / table_group["extrap"])

    table_group.round(rnd).to_csv(f"analysis/{save_name}_rmse.csv")
    print(f"File saved: analysis/{save_name}_rmse.csv")
    return table_group


def plot_finish_groups(test_data, label_pair, num=4, overall=True, save_name: str = "bos", 
                       palette="viridis", grouping="finish"): # grouping="age"
    """Create RMSE plot grouped by finish time"""
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(10)
    sns.set_palette(palette, n_colors=num, desat=0.8)
    colors1 = sns.color_palette()
    # colors1 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:num]
    colors2 = [to_rgba(c, alpha=0.3) for c in colors1]
    mixed_colors = [val for pair in zip(colors2, colors1) for val in pair]

    table_group = group_data(test_data, group_feat=grouping, num_groups=num, pref="Q", group_name="group", lbl="model2")
    table_group.plot(style='.-', width=0.8, #alpha=1, 
        color=mixed_colors, edgecolor="black", linewidth=0.3,
        ax=ax, legend=False, kind="bar")

    if overall:
        rmse = rmse_table(test_data, label_pair)
        plt.plot(range(8), rmse[label_pair[0]], color="black", alpha=0.4, linestyle=':',  marker=".", label='overall_extrap')
        plt.plot(range(8), rmse[label_pair[1]], color="black", alpha=0.4, linestyle='-',  marker=".", label='overall_model')
    # fig.patch.set_facecolor(('yellow', 0.05)) # This changes the grey to white
    # ax.set_facecolor(("orange", 0.05))
    plt.legend()
    # ax.legend(ncols=1, loc="upper right")
    plt.grid(alpha=0.8)
    plt.xticks(range(8), rotation=0)
    plt.yticks(range(0, int(table_group.max(axis=None) + 5), 5))
    plt.xlabel("Distance Into Race (km)")
    plt.ylabel("Prediction Error (RMSE), in minutes")
    plt.title("Average Error By Finish Groups")
    plt.savefig(f"analysis/{save_name}_rmse_groups.png", bbox_inches="tight")
    print(f"File saved: analysis/{save_name}_rmse_groups.png")
    sns.reset_defaults()
    plt.close()
    return table_group


def plot_finish_age_gender(test_data, label_pair, num=4, overall=True, 
                           save_name: str = "bos", palette="viridis", grouping="age"):
    """Create RMSE plot grouped by age group, for each gender"""
    fig, ax = plt.subplots(ncols=2, sharey=True)
    fig.set_figheight(12)
    fig.set_figwidth(24)
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]

    sns.set_palette(palette, n_colors=num, desat=0.8)
    colors1 = sns.color_palette()
    colors2 = [to_rgba(c, alpha=0.3) for c in colors1]
    mixed_colors = [val for pair in zip(colors2, colors1) for val in pair]

    for i, g in enumerate(["F", "M"]):
        test_data2 = test_data[test_data["male"] == i]
        table_group = group_data(test_data2, group_feat=grouping, num_groups=num, pref="AG", group_name="group", lbl="model2")
        table_group.plot(style='.-', width=0.8, #alpha=1, 
            color=mixed_colors,edgecolor="black", linewidth=0.3,
            ax=ax[i], legend=False, kind="bar")

        if overall:
            rmse = rmse_table(test_data2, label_pair)
            ax[i].plot(range(8), rmse[label_pair[0]], color="black", alpha=0.4, linestyle=':',  marker=".", label='overall_extrap')
            ax[i].plot(range(8), rmse[label_pair[1]], color="black", alpha=0.4, linestyle='-',  marker=".", label='overall_model')

        ax[i].set(xlabel="Distance Into Race (km)", ylabel="Prediction Error (RMSE), in minutes", 
            title=f"Average Error By Finish Groups - {g}") # , ylim=(0.2, 1)
    
        ax[i].legend()
        ax[i].grid(alpha=0.8)
        ax[i].set_xticklabels(mks, rotation=0)

    plt.savefig(f"analysis/{save_name}_rmse_gender_age.png", bbox_inches="tight")
    print(f"File saved: analysis/{save_name}_rmse_gender_age.png")
    sns.reset_defaults()
    plt.close()
    return

def plot_interval_checks(itbl: pd.DataFrame, pred_names: list, intervals: list = [50, 80, 95], 
                        linestyles = ["-.", "--.", ":."], save_name: str = "bos", rnd=3):
    """Plot the interval check and sizes for each model specified in pred_names"""
    fig, ax = plt.subplots(nrows=2)
    fig.set_figheight(18)
    fig.set_figwidth(12)

    n = len(pred_names)
    colors = [f"C{i}" for i in range(len(pred_names))] * len(intervals)
    mks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    tables0, tables1 = [], []
    for conf in intervals:
        
        sublabels = [f"{p}-size{conf}" for p in pred_names]
        result_tbl = itbl.groupby(["dist"])[sublabels].apply(lambda x: x.sum() / len(x)).loc[mks]
        tables0.append(result_tbl)

        sublabels = [f"{p}-in{conf}" for p in pred_names]
        result_tbl = itbl.groupby(["dist"])[sublabels].apply(lambda x: x.sum() / len(x)).loc[mks]
        tables1.append(result_tbl)

    big_table0, big_table1 = pd.concat(tables0, axis=1), pd.concat(tables1, axis=1)

    styles = []
    for s in linestyles:
        styles += [s] * n
    ax0 = big_table0.plot(label=big_table0.columns,  style=styles , linewidth=2, grid=True, alpha=0.75, color=colors, ax=ax[0])
    ax1 = big_table1.plot(label=big_table1.columns,  style=styles , linewidth=2, grid=True, alpha=0.75, color=colors, ax=ax[1])

    ax0.set(xlabel="Distance Into Race", ylabel="Credible Interval Sizes",
            title="Credible Interval Sizes") #, facecolor=("orange", 0.05)
    ax1.set(xlabel="Distance Into Race", ylabel="Proportion of Actual Finish Times Within Credible Interval", 
            title="Proportion Within Interval Over Different Distances") # , ylim=(0.2, 1)

    leg01 = ax0.legend([Line2D([0,1],[0,1],linestyle=s, color='black') for s in ["-.", "--", ":"]], ["50", "80", "95"], loc="upper center")
    ax[0].add_artist(leg01)
    leg02 = ax0.legend([Line2D([0,1],[0,1], color=c) for c in colors[:n]], pred_names, loc="upper right")
    ax[0].add_artist(leg02)

    leg11 = ax1.legend([Line2D([0,1],[0,1],linestyle=s, color='black') for s in ["-.", "--", ":"]], ["50", "80", "95"], loc="lower center")
    ax[1].add_artist(leg11)
    leg12 = ax1.legend([Line2D([0,1],[0,1], color=c) for c in colors[:n]], pred_names, loc="lower right")
    ax[1].add_artist(leg12)

    plt.savefig(f"analysis/{save_name}_intervals.png", bbox_inches="tight")
    print(f"File saved: analysis/{save_name}_intervals.png")
    plt.close()

    big_table0.round(rnd).to_csv(f"analysis/{save_name}_int_size.csv")
    print(f"File saved: analysis/{save_name}_int_sizes.csv")
    big_table1.round(rnd).to_csv(f"analysis/{save_name}_int_checks.csv")
    print(f"File saved: analysis/{save_name}_int_checks.csv")
    return big_table0, big_table1


if __name__ == "__main__":
    print('start')
    get_plot_dist()
    get_extrap_scatter("bos") # , mks=["5K", "15K", "25K", "35K"])
    get_extrap_scatter("nyc")
    get_extrap_scatter("chi")
    print('fin')