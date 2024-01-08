
import numpy as np
import time
import pandas as pd
from collections import defaultdict
from typing import Union
from bayes_model import _prior_dist, person_dict
from likelihoods import main_lk
from utils import get_training_set, get_test_set, round_df
import matplotlib.pyplot as plt
from gmm import quals, quals_test


def percentile(arr: np.array, q: float):
    """Return the index of the q percentile"""
    arr = arr / arr.sum()
    return np.argmax(np.cumsum(arr) >= q)


def in_interval(arr: np.array, actual: int, conf: float = 0.95):
    lower = (1 - conf) / 2
    upper = 1 - lower
    lower = percentile(arr, q=lower)
    upper = percentile(arr, q=upper)
    return (actual >= lower) and (actual <= upper)


def interval_size(arr: np.array,  conf: float = 0.95):
    lower = (1 - conf) / 2
    upper = 1 - lower
    lower = percentile(arr, q=lower)
    upper = percentile(arr, q=upper)
    return upper - lower


def _val_median(bayes_table: pd.DataFrame, person_actual: int, dist: str = "5K"):
    bayes_array = bayes_table[dist]
    return percentile(bayes_array, q=0.5) - person_actual


def _val_mode(bayes_table: pd.DataFrame, person_actual: int, dist: str = "5K"):
    bayes_array = bayes_table[dist]
    return np.argmax(bayes_array) - person_actual


def _val_conf_lists(bayes_table: pd.DataFrame, person_actual: int, dist: str = "5K"):
    bayes_array = bayes_table[dist]
    conf_levels = [.5, .9, .95]
    return [in_interval(bayes_array, actual=person_actual, conf=c) for c in conf_levels]


def _val_conf_lens(bayes_table: pd.DataFrame, person_actual: int, dist: str = "5K"):
    bayes_array = bayes_table[dist]
    conf_levels = [.5, .9, .95]
    return [interval_size(bayes_array,  conf=c) for c in conf_levels]


def _get_default_error_counts(df: pd.DataFrame, dist: str = "5K",):
    adjust = {"5K": 42.195/5, "10K": 42.195/10, "15K": 42.195/15, "20K": 42.195/20, "HALF": 2,
              "25K": 42.195/25, "30K": 42.195/30, "35K": 42.195/35, "40K": 42.195/40}
    default_error = (((df[dist] * adjust[dist]) - df["Finish Net"]) // 60).astype(int)
    default_error_counts = (default_error.value_counts() / len(default_error)).rename("default")
    return default_error_counts


def _get_model_error_counts(error_list: pd.DataFrame):
    model_error_counts = (error_list.value_counts() / len(error_list)).rename("model")
    return model_error_counts


def _comparison_table(
        error_counts_lists: dict,
        dist: str = "5K",
        save: Union[str, None] = None,
):
    names = list(error_counts_lists.keys())
    all_errors = pd.DataFrame(error_counts_lists[names[0]].rename(names[0]))
    for name in names[1:]:
        ls = pd.DataFrame(error_counts_lists[name].rename(name))
        all_errors = all_errors.merge(ls, left_index=True, right_index=True).sort_index()

    if isinstance(save, str):
        plt.plot(all_errors, label=all_errors.columns)
        plt.xlabel("Range of Error")
        plt.ylabel("Proportion of People Correctly Predicted")
        plt.title(f"Comparison over {dist}")
        plt.xlim(-40, 40)
        plt.legend()
        plt.savefig(save)
        plt.clf()

    return all_errors


def run_all_analyses(
        people_data: pd.DataFrame,
        functions: dict,
        marks: list,
        lk_dict: dict,
        informed=True,
        max_finish: int = 500
):
    good_data = []
    result_dict = {name: defaultdict(list) for name, func in functions.items()}
    # prior = _prior_dist(informed=informed, max_time=max_finish)
    prior = _prior_dist(informed=True, max_time=1001)
    for i in range(len(people_data)):
        try:
            person_info = people.iloc[i]
            table, actual = person_dict(person=person_info, checkpoints=marks, prior=prior, lk_tables=lk_dict)
            for mark in marks:
                for func_name, func in functions.items():
                    val = func(table, actual, mark)
                    result_dict[func_name][mark].append(val)

            good_data.append(person_info)
            if i % 1000 == 0:
                print('inside', i)
        except:
            print('failed', i)
            continue

    return result_dict, pd.DataFrame(good_data)


if __name__ == "__main__":
    s = time.time()
    max_fin = 1001
    train = get_training_set(pd.read_csv("processed_data/full_data_secs.csv"))  # quals
    train = train[train["Finish Net"] // 60 < 360]

    # train = train[train["M/F"] == "M"]
    marks = checkpoints = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    df = pd.read_csv("processed_data/full_data_secs.csv")  # quals_test  #
    people = round_df(get_test_set(df), marks)
    # people = people[people["M/F"] == "M"]
    lks = main_lk(df=train, marks_list=marks, store=False, process=False)

    print('load likelihoods:', time.time() - s)
    results, good_data = run_all_analyses(
        people_data=people,
        functions={
            "median": _val_median,
            "mode": _val_mode,
            "confs": _val_conf_lists,
            "conf_lens": _val_conf_lens},
        marks=checkpoints,
        lk_dict=lks,
        informed=True,
        max_finish=max_fin,
    )
    print(len(people))

    pd.DataFrame({m: pd.DataFrame(results['conf_lens'][m])[0] for m in marks}).to_csv('analysis/conf_lens_50.csv')
    pd.DataFrame({m: pd.DataFrame(results['conf_lens'][m])[1] for m in marks}).to_csv('analysis/conf_lens_90.csv')
    pd.DataFrame({m: pd.DataFrame(results['conf_lens'][m])[2] for m in marks}).to_csv('analysis/conf_lens_95.csv')
    abs(pd.DataFrame({m: results["mode"][m] for m in marks})).to_csv('mode.csv')
    abs(pd.DataFrame({m: results["median"][m] for m in marks})).to_csv('median.csv')

    mode_df = pd.DataFrame({mark: pd.Series(error_list) for mark, error_list in results["mode"].items()})
    median_df = pd.DataFrame({mark: pd.Series(error_list) for mark, error_list in results["median"].items()})

    print(time.time() - s)
    # mode_t = comparison_table(people2, mode_df, "5K", save="Compare5KMode.png")

    conf_table = pd.DataFrame([pd.DataFrame(results["confs"][i]).sum() / len(results['confs'][i]) for i in marks])

    mode_err_probs = pd.DataFrame({m: mode_df[m].value_counts().sort_index() / mode_df[m].count() for m in mode_df.columns}).fillna(0)
    median_err_probs = pd.DataFrame({m: median_df[m].value_counts().sort_index() / median_df[m].count() for m in median_df.columns}).fillna(0)
    # mode_err_count = pd.DataFrame({m: mode_df[m].value_counts().sort_index() for m in mode_df.columns}).fillna(0).astype(int)

    for dist in marks:
        d2 = get_test_set(df)
        d2 = d2[d2["M/F"] == "M"]
        def_probs = _get_default_error_counts(d2, dist)
        mode_probs = mode_err_probs[dist]
        median_probs = median_err_probs[dist]
        _comparison_table(
            error_counts_lists={"def": def_probs, "mode": mode_probs, "median": median_probs},
            dist="5K",
            save=f"CompareAll{dist}.png"
        )
    print('f')

# pd.DataFrame(person_dict(person=good_data.iloc[25709], checkpoints=marks, prior=_prior_dist(informed=True, max_time=500), lk_tables=lks)[0])
# person_dict(person=people.iloc[231], checkpoints=marks, prior=_prior_dist(informed=True, max_time=500), lk_tables=lks)[1]


# test = pd.DataFrame({'mode_diff': results["mode"]["5K"], '5K_bin': good_data["5K"], 'Fin': good_data["Finish Net"], 'pred': good_data["5K"] * 42.195 / 5})
# test["mode_pred"] = test["mode_diff"] + test["Fin"]
# test.groupby("5K_bin")[["mode_diff", "mode_pred", "Fin"]].agg(["mean", "count"])

