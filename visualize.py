
from typing import Union, List
import matplotlib.pyplot as plt
from utils import int_to_str_time, _get_marks, _prior_dist
# from gmm import quals_train
from bayes_model import *


def prior_compare(
        table1: dict,
        table2: dict,
        show_list: List[str],
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
    colors = plt.get_cmap()(np.linspace(0.2, 0.8, len(show_list)))
    tables = [table1, table2]
    axes = [ax1, ax2]
    prior = ["Prediction: Informed Prior", "Prediction: Uniform Prior"]

    for axis, table, p in zip(axes, tables, prior):
        table['index'] = range(actual - plot_range, actual + plot_range + 1)
        axis.plot(table['index'], table["0K"], label="prior", color="black")
        for dist, color in zip(show_list, colors):
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
        show_list: List[str],
        save: Union[str, None] = None,
        actual: Union[int, None] = None,
        cmap_str: str = 'YlOrRd',
        plot_range: Union[int, None] = None,
):
    """Plot the bayes predictions for a given runner"""
    plt.figure(figsize=(12, 10))

    plt.set_cmap(cmap_str)
    colors = plt.get_cmap()(np.linspace(0.2, 0.8, len(show_list)))

    if (isinstance(plot_range, int)) and (isinstance(actual, int)):
        bayes_dict = {k: v[actual - plot_range: actual + plot_range + 1] for k, v in bayes_dict.items()}
        bayes_dict['index'] = range(actual - plot_range, actual + plot_range + 1)
    else:
        bayes_dict['index'] = range(500)

    plt.plot(bayes_dict['index'], bayes_dict["0K"], label="prior", color="black")
    for dist, color in zip(show_list, colors):
        plt.plot(bayes_dict['index'], bayes_dict[dist], label=dist, color=color)

    plt.legend()
    plt.xlabel("Time (MM:SS)", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.yticks([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
    x_labels = plt.xticks()[0]
    plt.xticks(x_labels, [int_to_str_time(t) for t in x_labels])

    plt.title(f"{save} Live Prediction", fontsize=20)

    actual = int(actual)
    if isinstance(actual, int):
        plt.vlines(actual, 0, 1, linestyles="dashed", color="black", label="actual")

    if isinstance(save, str):
        plt.savefig(f"plots/Plot: {save}.png", facecolor='w')

    plt.close()


def plot_from_data(data: dict, name: str, marks: list, prior, lk_data, s2):
    if "0K" not in data.keys():
        data["0K"] = 0
    m = [m for m in marks if m in data.keys()]
    table = full_bayes_dict(data, prior, lk_data, s2)
    plot_bayes_dict(bayes_dict=table, show_list=m, save=f"test",
                    actual=table["Posterior"].argmax(), plot_range=30, cmap_str="inferno")
    return


if __name__ == '__main__':
    max_finish = 500
    df = pd.read_csv("processed_data/full_data_secs.csv")
    train_data, train_info = get_train_set(df, zero_k=True)
    store_initial_prior(finish=train_data[:, -1] // 1, max_time=max_finish)
    s2_matrix = get_s2_dict(train_data[:, -1], bin_mapping=np.ones(500), max_len=500)
    marks = _get_marks(marks_list=None, zero_k=True, finish=False)
    marks_w_fin = _get_marks(marks_list=marks, zero_k=True, finish=True)

    df = pd.read_csv("processed_data/nucr_runners.csv")
    test_data, test_info = get_test_set(df, zero_k=True)  # df = pd.read_csv("processed_data/nucr_runners.csv")
    print("start")
    uninformed_prior = _prior_dist(informed=False, max_time=max_finish)
    for i, row in enumerate(test_data):
        informed_prior = _prior_dist(informed=True, max_time=max_finish)
        dict1 = person_dict(person=row, marks=marks, prior=informed_prior, lk_data=train_data, s2=s2_matrix)[0]
        dict2 = person_dict(person=row, marks=marks, prior=uninformed_prior, lk_data=train_data, s2=s2_matrix)[0]
        print(i, test_info.iloc[i]["Name"])

        # prior_compare(dict1, dict2, marks, save=f"Compare: {test_info.iloc[i]['Name']}",
        #               actual=int(row[-1]), plot_range=60, cmap_str="inferno")

        plot_bayes_dict(bayes_dict=dict1, show_list=marks, save=f"{test_info.iloc[i]['Name']}",
                        actual=int(row[-1]), plot_range=60, cmap_str="inferno")

    testing = {"5K": 20, "10K": 40, "15K": 519}
    # plot_from_data(testing, name="test", marks=marks, prior=informed_prior, lk_data=train_data, s2=s2_matrix)
