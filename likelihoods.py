
# likelihoods.py: store, process, and read in level 1 or level 2 likelihoods

import json
import numpy as np
import pandas as pd
from utils import get_training_set


def new_fix_df(old_data: pd.DataFrame, last_dist, last_fix, curr_dist, curr_fix, fin_fix):
    """"""
    labels = {0: 0.5, 15: 0.25, 30: 0, 45: -0.25, 60: -0.5}
    marks_list = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    new_data = old_data.copy()
    new_data["0K"] = 0
    new_data[last_dist] = ((new_data[last_dist] + last_fix) // 60) + labels[last_fix]
    new_data[curr_dist] = ((new_data[curr_dist] + curr_fix) // 60) + labels[curr_fix]
    new_data["Finish Net"] = ((new_data["Finish Net"] + fin_fix) // 60) + labels[fin_fix]
    new_data = new_data[["0K"] + marks_list + ["Finish Net"]].round(2).astype(str)
    return new_data


def add_to_lk(old_data: pd.DataFrame, last_fix, curr_fix, fin_fix, all_lks: dict = {}):
    """Add likelihoods to dict using a different binning for the data df"""
    last_dict = {"5K": "0K", "10K": "5K", "15K": "10K", "20K": "15K", "HALF": "20K",
                 "25K": "HALF", "30K": "25K", "35K": "30K", "40K": "35K"}
    for dist, last_dist in last_dict.items():
        fixed_df = new_fix_df(
            old_data=old_data, last_dist=last_dist, last_fix=last_fix,
            curr_dist=dist, curr_fix=curr_fix, fin_fix=fin_fix
        )
        counts_data = fixed_df.groupby([last_dist, dist, "Finish Net"]).size().reset_index().values
        for last_mark, mark, fin, count in counts_data:
            last_mark, mark, fin, count = str(last_mark), str(mark), str(fin), str(count)
            if dist not in all_lks.keys():
                all_lks[dist] = {}
            if last_mark not in all_lks[dist].keys():
                all_lks[dist][last_mark] = {}
            if fin not in all_lks[dist][last_mark].keys():
                all_lks[dist][last_mark][fin] = {}
            all_lks[dist][last_mark][fin][mark] = count

    return all_lks


def _array_for_finish(group: pd.DataFrame, bound):
    # indexes = np.linspace(0, bound, 2 * bound + 1)
    lks = np.zeros(2 * bound + 1)

    for fin, mar, lk in group.values:
        lks[int(fin * 2)] = lk

    first = np.nonzero(lks)[0][0]
    return np.concatenate([[first], np.trim_zeros(lks)])


def process_lks(filename: str = "likelihoods/all_likelihoods.json"):
    """Process the stored likelihoods"""
    with open(filename) as file:
        dct = json.load(file)

    print('processing lks...')
    new_dict = {}
    for dist in dct.keys():
        new_dict[dist] = {}
        for last_mark in dct[dist].keys():
            counts_table = []
            for finish in dct[dist][last_mark].keys():
                for mark, count in dct[dist][last_mark][finish].items():
                    counts_table.append([finish, mark, count])

            counts = pd.DataFrame(counts_table, columns=["Fin", "Mar", "Lks"]).astype(float)
            counts["Lks"] = counts.groupby("Fin")["Lks"].transform(lambda x: x / x.sum())
            bound = int((counts["Fin"].max() + 1) // 1)
            c = counts.groupby("Mar").apply(lambda x: _array_for_finish(x, bound))

            last_mark = str(float(last_mark))
            new_dict[dist][last_mark] = {k: list(v) for k, v in c.to_dict().items()}
        print('processed: ', dist)
    for dist, dct in new_dict.items():
        filename = f"likelihoods/likelihoods_{dist}.json"
        with open(filename, "w") as file:
            json.dump(dct, file)
        print('loaded: ', dist)

    return


def read_likelihoods(marks_list: list):
    """Read in dict from file for each mark"""
    new_dict = {}
    for dist in marks_list:
        with open(f"likelihoods/likelihoods_{dist}.json") as file:
            new_dict[dist] = json.load(file)

    return new_dict


def main_lk(
        df: pd.DataFrame,
        marks_list: list,
        filename: str = "likelihoods/all_likelihoods.json",
        store: bool = True,
        process: bool = True,
):
    if store:
        print("storing lks...")
        all_lks = {}
        for last_fix in [30, 0]:
            for curr_fix in [30, 0]:
                for fin_fix in [30]:
                    all_lks = add_to_lk(
                        old_data=df, last_fix=last_fix, curr_fix=curr_fix,
                        fin_fix=fin_fix, all_lks=all_lks
                    )
                    print('added: ', last_fix, curr_fix, fin_fix)

        with open(filename, "w") as file:
            json.dump(all_lks, file)

        print('stored lks')
    if process:
        process_lks()

    all_lks = read_likelihoods(marks_list)
    return all_lks


def _return_lk(
        lk_dict: dict,
        dist: str,
        last_mark: str,
        curr_mark: str,
        max_len: int = 1001,
):
    """Return the likelihood array from the dictionary"""
    last_lk = lk_dict[dist].get(last_mark, {})
    lk_array = last_lk.get(curr_mark, np.ones(1001))

    array1 = np.zeros(int(lk_array[0]))
    array2 = np.array(lk_array[1:])
    array3 = np.zeros(max_len - len(array1) - len(array2))
    return np.concatenate([array1, array2, array3])


if __name__ == '__main__':
    data = get_training_set(pd.read_csv("processed_data/full_data_secs.csv"))  # data = quals
    marks = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K"]
    lks = main_lk(df=data, marks_list=marks, store=True, process=True)
    print('done')
