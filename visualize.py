import numpy as np
import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from bayes_model import person_dict, _prior_dist, read_in_jsons, get_training_set
from utils import int_to_str_time


def prior_compare(
        table1: dict,
        table2: dict,
        show: List[str],
        save: Union[str, None] = None,
        cmap_str: str = 'YlOrRd',
        actual: Union[int, None] = None,
        plot_range: Union[int, None] = None,
):
    """Plot the bayes predictions for both the informed prior and the uniform prior"""
    if (isinstance(plot_range, int)) and (isinstance(actual, np.int64)):
        table1 = {k: v[actual - plot_range: actual + plot_range + 1] for k, v in table1.items()}
        table2 = {k: v[actual - plot_range: actual + plot_range + 1] for k, v in table2.items()}
    f, (ax1, ax2) = plt.subplots(1, 2, sharey="all", figsize=(34, 18))

    plt.set_cmap(cmap_str)
    colors = plt.get_cmap()(np.linspace(0.1, 0.8, len(show)))
    tables = [table1, table2]
    axes = [ax1, ax2]
    prior = ["Prediction: Informed Prior", "Prediction: Uniform Prior"]

    for axis, table, p in zip(axes, tables, prior):
        table['index'] = range(actual - plot_range, actual + plot_range + 1)
        axis.plot(table['index'], table["0K"], label="prior", color="black")
        for dist, color in zip(show, colors):
            axis.plot(table['index'], table[dist], label=dist, color=color)

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


def plot_bayes_dict(
        bayes_dict: dict,
        show: List[str],
        save: Union[str, None] = None,
        actual: Union[int, None] = None,
        cmap_str: str = 'YlOrRd',
        plot_range: Union[int, None] = None,
):
    """Plot the bayes predictions for a given runner"""
    plt.figure(figsize=(12, 12))

    plt.set_cmap(cmap_str)
    colors = plt.get_cmap()(np.linspace(0.1, 0.8, len(show)))

    if (isinstance(plot_range, int)) and (isinstance(actual, np.int64)):
        bayes_dict = {k: v[actual - plot_range: actual + plot_range + 1] for k, v in bayes_dict.items()}
        bayes_dict['index'] = range(actual - plot_range, actual + plot_range + 1)
    else:
        bayes_dict['index'] = range(500)

    plt.plot(bayes_dict['index'], bayes_dict["0K"], label="prior", color="black")
    for dist, color in zip(show, colors):
        plt.plot(bayes_dict['index'], bayes_dict[dist], label=dist, color=color)

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


if __name__ == '__main__':
    data = get_training_set(pd.read_csv("processed_data/full_data_mins.csv"))
    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    lks = read_in_jsons(marks)
    people = pd.read_csv("processed_data/nucr_runners.csv", index_col=0)
    show = marks
    max_finish = 500

    print(len(people))
    for i in range(len(people)):
        person_info = people.iloc[i]
        informed_prior = _prior_dist(informed=True, max_time=max_finish)
        table_i = person_dict(person=person_info, checkpoints=marks, prior=informed_prior, lk_tables=lks)[0]
        uninformed_prior = _prior_dist(informed=False, max_time=max_finish)
        table_u = person_dict(person=person_info, checkpoints=marks, prior=uninformed_prior, lk_tables=lks)[0]

        prior_compare(table_i, table_u, marks, save=f"Compare: {person_info['Name']}",
                      actual=person_info["Finish Net"], plot_range=60, cmap_str="inferno")

        plot_bayes_dict(bayes_dict=table_u, show=marks, save=f"Plot: {person_info['Name']}",
                        actual=person_info["Finish Net"], plot_range=20, cmap_str="inferno")

        # plot_person(people.iloc[0], lk_dict=likelihoods, name="Vinny", show=checkpoints)
