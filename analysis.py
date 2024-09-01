
import random
import numpy as np
import pandas as pd
from reg_quart import quart_reg
from utils import int_to_str_time, get_data, get_models_and_traces

random.seed(2024)
train, test = get_data(filepath="processed_data/full_data_secs.csv", size_train=500, size_test=400)
_, nucr_data = get_data(filepath="processed_data/nucr_runners.csv", size_train=0, size_test=9)

train2 = train[train["dist"] != "5K"]
test2 = test[test["dist"] != "5K"]
marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
[[model1, model2], [trace1, trace2]] = get_models_and_traces()


y_true = (42195 / 60) / test["finish"]
b1 = model1.prediction(test, trace1)
b2 = model2.prediction(test, trace2)

test["bayes1"] = np.abs(b1.mean(axis=1) - y_true)
test["bayes2"] = np.abs(b2.mean(axis=1) - y_true)

q1 = quart_reg(train, test, formula="finish ~ total_pace + dist")
q2 = quart_reg(train, test, formula="finish ~ total_pace + curr_pace + dist")

test["quant1"] = np.abs(q1 - y_true)
test["quant2"] = np.abs(q2 - y_true)



###########
# from old analysis.py
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

###########