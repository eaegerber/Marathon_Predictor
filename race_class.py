
from typing import List
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import str_to_int_time, int_to_str_time, time_to_pace, conv1


class RaceSplits():

    marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]

    def __init__(self):
        self.curr_dist = "0K"
        self.total_pace = 0
        self.curr_pace = 0
        self.curr_time: int = 0
        self.stored_paces = []

    def next_dist(self):
        new_dist = {"0K": "5K", "5K": "10K", "10K": "15K", "15K": "20K", "20K": "25K", "25K": "30K", 
                     "30K": "35K", "35K": "40K", "40K": "Finish", "Finish": "Finish"}
        
        self.curr_dist = new_dist[self.curr_dist]
        return self.curr_dist


    def show_next_dist(self):
        new_dist = {"0K": "5K", "5K": "10K", "10K": "15K", "15K": "20K", "20K": "25K", "25K": "30K", 
                     "30K": "35K", "35K": "40K", "40K": "Finish", "Finish": "Finish"}
        
        return new_dist[self.curr_dist]

    
    def get_paces(self):
        return self.curr_dist, self.curr_pace, self.total_pace
    
    def update_pace(self, time: str):
        self.next_dist()
        last_time = self.curr_time
        num_time = str_to_int_time(time)
        assert num_time > last_time, "Next time should be larger than previous time"
        self.curr_time = num_time
        self.total_pace = conv1[self.curr_dist] / self.curr_time
        # print('inside: ', self.curr_time, last_time)
        diff = (self.curr_time - last_time)
        self.curr_pace = conv1["5K"] / diff
        self.stored_paces.append(self.get_paces())
        # self.next_dist()  # update curr_dist

    def update_all_splits(self, times: List[str]):
        for time in times:
            self.update_pace(time)

    def get_stored_paces(self):
        return pd.DataFrame(self.stored_paces, columns=["dist", "curr_pace", "total_pace"])
    
    def get_person_dict(self):
        person = self.get_stored_paces()
        person[["total_pace", "curr_pace"]] = time_to_pace(person[["total_pace", "curr_pace"]], "Finish Net")
        person_dict = {row["dist"]: (row["dist"], row["total_pace"], row["curr_pace"]) for _, row in person.iterrows()}
        # print('a', person_dict)
        return person_dict
    

    def posterior_array(self, models: list, traces: list, show: list = ["10K", "15K"]):
        info = self.get_stored_paces()
        info = info[info["dist"].isin(show)]

        model1, model2 = models
        trace1, trace2 = traces


        first_split = info[info["dist"] == "5K"]
        # return model1.prediction(info, trace1)
        if first_split.shape[0] == 1:
            array1 = model1.prediction(first_split, trace1, progressbar=False)

        other_splits = info[info["dist"] != "5K"]
        if other_splits.shape[0] > 0:
            array2 = model2.prediction(other_splits, trace2, progressbar=False)
        else:
            return array1

        if first_split.shape[0] != 1:
            return array2
        
        return np.concatenate([array1, array2])


# if __name__ == "__main__":
#     marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
#     print(marks.index("15K"))

#     rs = RaceSplits()
#     # print('paces', rs.get_paces())
#     print('curr_dist', rs.curr_dist, rs.show_next_dist())
#     rs.update_pace("15:45")

#     # print('paces', rs.get_paces())
#     print('curr_dist', rs.curr_dist, rs.show_next_dist())
#     rs.update_pace("32:12")
#     rs.update_pace("49:57")
#     rs.get_person_dict()
#     rs.update_pace("64:45")
#     rs.update_pace("64:46")
#     # rs.update_pace("63:45")
#     rs.get_person_dict()
#     print('s', rs.get_stored_paces())


def table_info(info: pd.DataFrame, show = ["5K", "10K"]):
    percentiles = np.percentile(info, [2.5, 10, 25, 50, 75, 90, 97.5], axis=1)
    percentile_names = ["lower95", "lower80", "lower50", "mean", "upper50", "upper80", "upper95"]
    table = pd.DataFrame(percentiles, index=percentile_names, columns=show)
    for col in table:
        table[col] = table[col].apply(lambda x: int_to_str_time(x))
    return table


def get_from_info(
    race: RaceSplits,
    models: list,
    traces: list,
    name: str = "",
    actual=None,
    show: list = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
):
    """Returns figure and table"""
    marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    shows = [m for m in marks if m in list(race.get_stored_paces()["dist"])]
    shows = [m for m in shows if m in show]
    # print('d', shows)
    fig = plt.figure(figsize=(12, 10))

    p_array = race.posterior_array(models, traces, shows)
    percentile_info = table_info(p_array, show=shows)
    
    table = []
    colors = plt.get_cmap()(np.linspace(0.2, 0.8, len(shows)))
    for i, dist in enumerate(shows):
        info = percentile_info[dist]
        info_strings = (dist, f'{info["mean"]}', f'{info["lower50"]}-{info["upper50"]}', f'{info["lower80"]}-{info["upper80"]}', f'{info["lower95"]}-{info["upper95"]}')
        table.append(info_strings)
        sns.kdeplot(p_array[i], color=colors[i], label=dist)


    x_labels = plt.xticks()[0]
    plt.xticks(x_labels, [int_to_str_time(t) for t in x_labels])
    plt.xlabel("Time (HH:MM)", fontsize=15)
    plt.ylabel("Probability", fontsize=15)
    plt.title(f"{name} Live Prediction", fontsize=17)
    if actual:
        plt.vlines(actual, 0, plt.yticks()[0].max(), linestyles="dashed", color="black", label="actual")

    plt.legend()
    print('done')
    return fig, pd.DataFrame(table, columns=["dist", "median", "range_50", "range_80", "range_95"])


def get_race_for_person(num, train_data,) -> RaceSplits:
    mlis = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    # train_data = pd.read_csv("processed_data/nucr_runners.csv")
    person = train_data.iloc[num]
    times = list(person[mlis].apply(lambda x: int_to_str_time(x)))
    actual = person["Finish Net"] / 60

    person_race = RaceSplits()
    person_race.update_all_splits(times)
    return person_race, actual

