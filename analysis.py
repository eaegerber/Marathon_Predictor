
import numpy as np
import time
import pandas as pd
from typing import Union
from collections import defaultdict
from bayes_model import _prior_dist, person_dict, initialize
import matplotlib.pyplot as plt
# from gmm import quals_train, quals_test


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


def _val_median(bayes_table: pd.DataFrame, person_actual: int, dst: str, mark: float):
    bayes_array = bayes_table[dst]
    return percentile(bayes_array, q=0.5) - person_actual


def _val_mode(bayes_table: pd.DataFrame, person_actual: int, dst: str, mark: float):
    bayes_array = bayes_table[dst]
    return np.argmax(bayes_array) - person_actual


def _val_mean(bayes_table: pd.DataFrame, person_actual: int, dst: str, mark: float):
    bayes_array = bayes_table[dst]
    return (np.dot(bayes_array, range(bayes_array.shape[0])) // 1) - person_actual


def _val_def(bayes_table: pd.DataFrame, person_actual: int, dst: str, mark: float):
    adjust = {"5K": 42.195 / 5, "10K": 42.195 / 10, "15K": 42.195 / 15, "20K": 42.195 / 20, "HALF": 2,
              "25K": 42.195 / 25, "30K": 42.195 / 30, "35K": 42.195 / 35, "40K": 42.195 / 40}
    mult = adjust[dst]
    return ((mark * mult) // 1) - person_actual


def _val_conf_lists(bayes_table: pd.DataFrame, person_actual: int, dst: str, mark: float):
    bayes_array = bayes_table[dst]
    conf_levels = [.5, .9, .95]
    return [in_interval(bayes_array, actual=person_actual, conf=c) for c in conf_levels]


def _val_conf_lens(bayes_table: pd.DataFrame, person_actual: int, dst: str, mark: float):
    bayes_array = bayes_table[dst]
    conf_levels = [.5, .9, .95]
    return [interval_size(bayes_array,  conf=c) for c in conf_levels]


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
        plt.xlabel("Error: Predicted - Actual (mins)")
        plt.ylabel("Proportion of People")
        plt.title(f"Prediction Error Comparison: {dist}")
        plt.xlim(-40, 40)
        plt.grid()
        plt.legend()
        plt.savefig(save)
        plt.clf()

    return all_errors


def run_all_analyses(
        test_df: np.array,
        functions: dict,
        marks: list,
        lk_data: np.array,
        informed=True,
        max_finish: int = 500
):
    good_data = []
    result_dict = {name: defaultdict(list) for name, func in functions.items()}
    prior = _prior_dist(informed=True, max_time=max_finish)
    for i, row in enumerate(test_df):
        try:
            person_info = row
            table, actual = person_dict(person=row, marks=marks, prior=prior, lk_data=lk_data, s2=s2_matrix)
            for j, dis in enumerate(marks):
                if dis == "0K":
                    continue
                for func_name, func in functions.items():
                    val = func(table, actual // 1, dis, row[j])
                    result_dict[func_name][dis].append(val)

            good_data.append(person_info)
            if i % 100 == 0:
                print('inside', i)
        except:
            print('failed', i)
            continue

    return result_dict, pd.DataFrame(good_data)


if __name__ == "__main__":
    f = time.time()
    sample_size = 1000
    # fname = "processed_data/full_data_women.csv"
    train_data, train_info, test_data, test_info, marks, s2_matrix, max_finish = initialize(
        # train_file=fname,
        # test_file=fname
    )
    # train_data, train_info, test_data, test_info, marks, s2_matrix, max_finish = initialize()
    # test_sample = test_data[np.random.choice(range(test_data.shape[0]), sample_size)]
    results, good_data = run_all_analyses(
        test_df=test_data,
        functions={
            "median": _val_median,
            "mode": _val_mode,
            "mean": _val_mean,
            "def": _val_def,
            "confs": _val_conf_lists,
            "conf_lens": _val_conf_lens},
        marks=marks,
        lk_data=train_data,
        informed=True,
        max_finish=max_finish,
    )

    marks = marks[1:]
    pd.DataFrame({m: pd.DataFrame(results['conf_lens'][m])[0] for m in marks}).to_csv('analysis/conf_lens_50.csv')
    pd.DataFrame({m: pd.DataFrame(results['conf_lens'][m])[1] for m in marks}).to_csv('analysis/conf_lens_90.csv')
    pd.DataFrame({m: pd.DataFrame(results['conf_lens'][m])[2] for m in marks}).to_csv('analysis/conf_lens_95.csv')
    pd.DataFrame({m: results["mode"][m] for m in marks}).to_csv('analysis/mode.csv')
    pd.DataFrame({m: results["median"][m] for m in marks}).to_csv('analysis/median.csv')
    pd.DataFrame({m: results["mean"][m] for m in marks}).to_csv('analysis/mean.csv')
    pd.DataFrame({m: results["def"][m] for m in marks}).to_csv('analysis/def.csv')

    mode_df = pd.DataFrame({mark: pd.Series(error_list) for mark, error_list in results["mode"].items()})
    median_df = pd.DataFrame({mark: pd.Series(error_list) for mark, error_list in results["median"].items()})
    mean_df = pd.DataFrame({mark: pd.Series(error_list) for mark, error_list in results["mean"].items()})
    def_df = pd.DataFrame({mark: pd.Series(error_list) for mark, error_list in results["def"].items()})

    conf_table = pd.DataFrame([pd.DataFrame(results["confs"][i]).sum() / len(results['confs'][i]) for i in marks])

    mode_err_probs = pd.DataFrame(
        {m: mode_df[m].value_counts().sort_index() / mode_df[m].count() for m in mode_df.columns}).fillna(0)
    median_err_probs = pd.DataFrame(
        {m: median_df[m].value_counts().sort_index() / median_df[m].count() for m in median_df.columns}).fillna(0)
    mean_err_probs = pd.DataFrame(
        {m: mean_df[m].value_counts().sort_index() / mean_df[m].count() for m in mean_df.columns}).fillna(0)
    def_err_probs = pd.DataFrame(
        {m: def_df[m].value_counts().sort_index() / def_df[m].count() for m in def_df.columns}).fillna(0)
    # mode_err_count = pd.DataFrame({m: mode_df[m].value_counts().sort_index() for m in mode_df.columns}).fillna(
    #     0).astype(int)

    for i, dist in enumerate(marks):
        if dist == "0K":
            continue
        mode_probs = mode_err_probs[dist]
        median_probs = median_err_probs[dist]
        mean_probs = mean_err_probs[dist]
        def_probs = def_err_probs[dist]
        _comparison_table(
            error_counts_lists={
                "def": def_probs,
                "mode": mode_probs,
                "median": median_probs,
                "mean": mean_probs},
            dist=dist,
            save=f"analysis/Compare{dist}.png"
        )
    print('f', time.time() - f)
    print('a')
