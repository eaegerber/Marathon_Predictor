import numpy as np
import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from bayes_model import person_table, _prior_dist
from utils import int_to_str_time  # , get_test_set_people


def prior_compare(
        table1: pd.DataFrame,
        table2: pd.DataFrame,
        show: List[str],
        save: Union[str, None] = None,
        cmap_str: str = 'YlOrRd',
        actual: Union[int, None] = None,
        plot_range: Union[int, None] = None,
):
    """Plot the bayes predictions for both the informed prior and the uniform prior"""
    if (isinstance(plot_range, int)) and (isinstance(actual, np.int64)):
        table1 = table1[actual - plot_range: actual + plot_range + 1]
        table2 = table2[actual - plot_range: actual + plot_range + 1]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey="all", figsize=(34, 18))

    plt.set_cmap(cmap_str)
    colors = plt.get_cmap()(np.linspace(0, .9, len(show)))
    tables = [table1, table2]
    axes = [ax1, ax2]
    prior = ["Prediction: Informed Prior", "Prediction: Uniform Prior"]

    for axis, table, p in zip(axes, tables, prior):
        axis.plot(table.index, table["0K"], label="prior", color="black")
        for dist, color in zip(show, colors):
            axis.plot(table.index, table[dist], label=dist, color=color)

        axis.legend(prop={'size': 20})
        axis.set_xlabel("Time (MM:SS)", size=30)
        axis.set_ylabel("Probability", size=30)
        plt.yticks([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
        x_labels = axis.get_xticks()
        axis.set_xticks(x_labels, [int_to_str_time(t) for t in x_labels])

        axis.set_title(f"{p}", size=40)

        if isinstance(actual, np.int64):
            axis.vlines(actual, 0, 1, linestyles="dashed", color="black", label="actual")

    if isinstance(save, str):
        plt.savefig(f"plots/{save}.png", facecolor='w')

    plt.close()


def plot_bayes_table(
        bayes_table: pd.DataFrame,
        show: List[str],
        save: Union[str, None] = None,
        actual: Union[int, None] = None,
        cmap_str: str = 'YlOrRd',
        plot_range: Union[int, None] = None,
):
    """Plot the bayes predictions for a given runner"""
    plt.figure(figsize=(16, 12))

    plt.set_cmap(cmap_str)
    colors = plt.get_cmap()(np.linspace(0, .9, len(show)))

    if (isinstance(plot_range, int)) and (isinstance(actual, np.int64)):
        bayes_table = bayes_table[actual - plot_range: actual + plot_range + 1]

    plt.plot(bayes_table.index, bayes_table["0K"], label="prior", color="black")
    for dist, color in zip(show, colors):
        plt.plot(bayes_table.index, bayes_table[dist], label=dist, color=color)

    plt.legend()
    plt.xlabel("Time (MM:SS)")
    plt.ylabel("Probability")
    plt.yticks([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    x_labels = plt.xticks()[0]
    plt.xticks(x_labels, [int_to_str_time(t) for t in x_labels])

    plt.title(f"{save} Live Prediction-")

    actual = int(actual)
    if isinstance(actual, int):
        plt.vlines(actual, 0, 1, linestyles="dashed", color="black", label="actual")

    if isinstance(save, str):
        plt.savefig(f"plots/{save}.png", facecolor='w')

    plt.close()


# def plot_person(
#         person_series: pd.Series,
#         lk_dict: dict,
#         name: str,
#         show: list,
#
# ):
#     """Plot bayes predictions for a person"""
#     table, actual = person_to_table(
#         person_series=person_series,
#         person_checkpoints=checkpoints,
#         lk_tables=lk_dict,
#         informed_prior=False,
#     )
#     plot_bayes_table(bayes_table=table, show=show,  actual=actual, save=name, cmap_str="inferno")
#     return


if __name__ == '__main__':
    people = pd.read_csv("processed_data/nucr_runners.csv", index_col=0)
    # people = test_people(pd.read_csv("processed_data/full_data_mins.csv"))
    df = pd.read_csv("processed_data/full_data_mins.csv")

    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    lk_dict = {mark: pd.read_csv(f"likelihood_tables/likelihood_{mark}.csv") for mark in marks}
    max_finish = 500

    for i in range(len(people)):  # for i in range(25000, 25006):
        person_info = people.iloc[i]

        informed_prior = _prior_dist(informed=True, max_time=max_finish)
        table1 = person_table(person=person_info, checkpoints=marks, prior=informed_prior, lk_tables=lk_dict)[0]

        uninformed_prior = _prior_dist(informed=False, max_time=max_finish)
        table2 = person_table(person=person_info, checkpoints=marks, prior=uninformed_prior, lk_tables=lk_dict)[0]

        # prior_compare(table1, table2, marks, save=f"Compare: {person_info['Name']}",
        #               actual=person_info["Finish Net"], plot_range=60, cmap_str="inferno")

        plot_bayes_table(bayes_table=table2, show=marks, save=f"Plot: {person_info['Name']}",
                         actual=person_info["Finish Net"], plot_range=40, cmap_str="inferno")
    # plot_person(people.iloc[0], lk_dict=likelihoods, name="Vinny", show=checkpoints)
