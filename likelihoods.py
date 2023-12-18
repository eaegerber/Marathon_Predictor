
# likelihoods.py: store, process, and read in level 1 or level 2 likelihoods

import json
import numpy as np
import pandas as pd


def store_likelihoods(dists: list, data: pd.DataFrame, level: int = 2):
    """Store likelihoods in a single json"""
    all_json = {}
    if level == 2:
        filename = "likelihoods/level2/all_likelihoods.json"
        last_dict = {"10K": "5K", "15K": "10K", "20K": "15K", "HALF": "20K",
                     "25K": "HALF", "30K": "25K", "35K": "30K", "40K": "35K"}

        dists = [d for d in dists if d != "5K"]
        for dist in dists:
            dist_json = {}
            last_dist = last_dict[dist]
            counts_data = data.groupby([last_dist, dist, "Finish Net"])["Name"].count().reset_index().values
            for last_mark, mark, fin, count in counts_data:
                last_mark, mark, fin, count = int(last_mark), int(mark), int(fin), int(count)
                if last_mark not in dist_json.keys():
                    dist_json[last_mark] = {}
                if fin not in dist_json[last_mark].keys():
                    dist_json[last_mark][fin] = {}
                dist_json[last_mark][fin][mark] = count

            all_json[dist] = dist_json
    else:
        filename = "likelihoods/level1/all_likelihoods.json"
        for dist in dists:
            dist_json = {}
            counts_data = data.groupby([dist, "Finish Net"])["Name"].count().reset_index().values
            for mark, fin, count in counts_data:
                mark, fin, count = int(mark), int(fin), int(count)
                if fin not in dist_json.keys():
                    dist_json[fin] = {}
                dist_json[fin][mark] = count

            all_json[dist] = dist_json

    with open(filename, "w") as file:
        json.dump(all_json, file)

    return


def process_likelihoods(max_length: int = 500, level: int = 2):
    """Process the stored likelihoods"""
    if level == 2:
        filename = "likelihoods/level2/all_likelihoods.json"
    else:
        filename = "likelihoods/level1/all_likelihoods.json"

    with open(filename) as file:
        dct = json.load(file)

    if level == 2:
        print('second')
        new_dict = {}
        for dist in dct.keys():
            new_dict[dist] = {}
            for last_mark in dct[dist].keys():
                lk_table2 = {}
                for finish in dct[dist][last_mark].keys():
                    for mark, count in dct[dist][last_mark][finish].items():
                        if int(finish) < max_length:
                            if int(mark) not in lk_table2.keys():
                                lk_table2[int(mark)] = np.zeros(shape=max_length, dtype=int)

                            lk_table2[int(mark)][int(finish)] = int(count)

                t = pd.DataFrame(lk_table2)
                lk_tbl2 = pd.DataFrame(t.T / t.sum(axis=1)).fillna(0).T
                new_dict[dist][last_mark] = {i: list(lk_tbl2[i]) for i in lk_tbl2.columns}  # new_lognorm_lk

        for dist, dct in new_dict.items():
            filename = f"likelihoods/level2/likelihoods_{dist}.json"
            with open(filename, "w") as file:
                json.dump(dct, file)
            print('loaded: ', dist)

        return
    else:
        print('first')
        new_dict = {}
        for dist in dct.keys():
            lk_table = np.zeros(shape=(max_length, max_length), dtype=int)
            for finish in dct[dist].keys():
                for mark, count in dct[dist][finish].items():
                    if int(finish) < max_length:
                        lk_table[int(finish)][int(mark)] = int(count)

            fin_counts = lk_table.sum(axis=1)
            lk_tbl = (lk_table.T / fin_counts).T
            lk_tbl2 = pd.DataFrame(lk_tbl).fillna(0)

            new_dict[dist] = {i: list(lk_tbl2[i]) for i in range(max_length)}  # new_lognorm_lk

        for dist, dct in new_dict.items():
            filename = f"likelihoods/level1/likelihoods_{dist}.json"
            with open(filename, "w") as file:
                json.dump(dct, file)
            print('loaded: ', dist)

        return


def read_likelihoods(marks: list, level: int = 2):
    """Read in dict from file for each mark"""
    new_dict = {}
    if level == 2:
        for dist in marks:
            if dist != "5K":
                with open(f"likelihoods/level2/likelihoods_{dist}.json") as file:
                    new_dict[dist] = json.load(file)

        with open(f"likelihoods/level1/likelihoods_5K.json") as file:
            new_dict["5K"] = json.load(file)
    else:
        for dist in marks:
            with open(f"likelihoods/level1/likelihoods_{dist}.json") as file:
                new_dict[dist] = json.load(file)

    return new_dict
