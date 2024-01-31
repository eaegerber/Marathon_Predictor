from shiny import render, ui
from shiny.express import input

ui.panel_title("Quantifying Uncertainty in Live Marathon Finish Time Predictions")
# ui.input_slider("n", "N", 0, 100, 10)

import numpy as np
import pandas as pd
from bayes_model import initialize, _prior_dist
from visualize import plot_from_data

nucr_filename = "processed_data/nucr_runners.csv"
nucr = pd.read_csv(nucr_filename)
train_data, train_info, test_data, test_info, marks, s2_matrix, max_finish = initialize(test_file=nucr_filename)
test_sample = test_data[np.random.choice(range(test_data.shape[0]), 10)]
informed_prior = _prior_dist(informed=True, max_time=max_finish)

ui.input_selectize("var", "Select Runner", choices=list(test_info["Name"]))
ui.input_selectize("var2", "Select Splits", choices=[str(marks[1:i+2]) for i in range(len(marks)-1)])
# ui.input_selectize("var", "Select variable", choices=["0", "1", "2", "3", "4", "5", "6"])


@render.text
def txt():
    return f"n*2 is {input.n() * 2}"


@render.plot
def hist():
    map = {name: i for i, name in enumerate(test_info["Name"])}
    name = input.var()
    i = map[name]
    row = test_data[i]
    testing = {dist: mark for dist, mark in zip(marks, row)}

    map2 = {str(marks[1:i+2]): marks[1:i+2] for i in range(len(marks)-1)}
    marks2 = map2[input.var2()]
    plot_from_data(testing, name=test_info.iloc[i]["Name"], marks=marks2, prior=informed_prior,
                   lk_data=train_data, s2=s2_matrix, plot_range=40)

    # df = load_penguins()
    # df[input.var()].hist(grid=False)
    # plt.xlabel(input.var())
    # plt.ylabel("count")

