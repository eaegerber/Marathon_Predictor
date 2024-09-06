
import faicons as fa
from shiny import reactive
from shiny.express import input, render, ui

import pandas as pd
from utils import int_to_str_time, get_model_and_trace
from race_class import RaceSplits, get_from_info, get_race_for_person

model1, trace1 = get_model_and_trace()

nucr_filename = "processed_data/nucr_runners.csv"
nucr = pd.read_csv(nucr_filename)
names = nucr["Name"]
marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
race_splits = RaceSplits()
ui.page_opts(title="Quantifying Uncertainty in Live Marathon Finish Time Predictions", fillable=True)

with ui.nav_panel("My Plot"):

    "View your finish time estimates in real time! Sequentially input your race splits (formatted MM:SS or HH:MM:SS) in the text box below, click go, and view your live prediction at that stage of the race. Input your times in 5km increments [5K, 10K, ..., 40K].  NOTE: if the app errors, refresh the page and try again."
    @render.text 
    @reactive.event(input.bttn)       
    def text1():     
        return f"Updated splits through {race_splits.show_next_dist()}"
    
    ui.input_text("runner_split1", "", "MM:SS")
    ui.input_action_button("bttn", "Go", class_="btn-success")
    # ui.input_action_button("bttn2", "Reset", class_="btn-success")
    # ui.input_checkbox_group("splits_list1", "Select Splits", choices=marks, selected=marks, inline=True)

    with ui.layout_columns(col_widths=[6, 6, 12]):

        @reactive.calc
        def get_info1():
            # print('hi', race_splits.show_next_dist())
            race_splits.update_pace(input.runner_split1())
            fig, table = get_from_info(race_splits, model1=model1, trace1=trace1, show=marks)                                       
            # ui.update_text("runner_split1", f"Input split for {race_splits.show_next_dist()}", 0)
            return fig, table
                                     
        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Table")

            with ui.popover(title="Filter credible intervals", placement="top"):        
                fa.icon_svg("gear")
                ui.input_checkbox_group("intervals1", "Credible Intervals", choices=["range_50", "range_80", "range_95"], 
                                        selected=["range_50", "range_95"], inline=True)
            
            @render.data_frame
            @reactive.event(input.bttn)
            def table1():
                info_table = get_info1()[1]
                return info_table[["dist", "median"] + list(input.intervals1())]

        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Plot")

            @render.plot
            @reactive.event(input.bttn)
            def nucr_hist1():
                return get_info1()[0]


with ui.nav_panel("NUCR Plots"):
    "View the plots of NUCR runners! The motivation behind this project involves the Northeastern Club Running team, which has dozens of Boston Marathon qualifiers every year. Here are some select NUCR runners that ran in the 2023 Boston Marathon race. The vertical dotted black line in the plot shows their actual finish time."
    ui.input_selectize("runner_name0", "Select Runner", choices=list(names))
    ui.input_checkbox_group("splits_list0", "Select Splits", choices=marks, selected=marks, inline=True)

    with ui.layout_columns(col_widths=[6, 6, 12]):

        @reactive.calc
        def get_info0():
            mapping = {name: i for i, name in enumerate(names)}
            name = input.runner_name0()
            i = mapping[name]

            race, actual = get_race_for_person(i, nucr)
            fig, table = get_from_info(race, model1=model1, trace1=trace1, show=input.splits_list0(), actual=actual)
            return fig, table, actual
                                     
        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Table")

            @render.text        
            def text0():     
                actual_time = get_info0()[2]
                return f"Actual finish time: {int_to_str_time(actual_time)}" #.txt()
            
            with ui.popover(title="Filter credible intervals", placement="top"):        
                fa.icon_svg("gear")
                ui.input_checkbox_group("intervals0", "Credible Intervals", choices=["range_50", "range_80", "range_95"], 
                                        selected=["range_50", "range_95"], inline=True)
            
            @render.data_frame
            def table0():
                info_table = get_info0()[1]
                return info_table[["dist", "median"] + list(input.intervals0())]

        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Plot")

            @render.plot
            def nucr_hist0():
                return get_info0()[0]
            

with ui.nav_panel("Project Info"):
    "Quantifying Uncertainty in Marathon Finish Time Predictions: In the middle of a marathon, a runner’s expected finish time is commonly estimated by extrapolating the average pace covered so far, assuming it to be constant for the rest of the race. These predictions have two key issues: the estimates do not consider the in-race context that can determine if a runner is likely to finish faster or slower than expected, and the prediction is a single point estimate with no information about uncertainty. We implement two approaches to address these issues: Bayesian linear regression and quantile regression. Both methods incorporate information from all splits in the race and allow us to quantify uncertainty around the predicted finish times. We utilized 15 years of Boston Marathon data (312,805 runners total) to evaluate and compare both approaches. Finally, we developed an app for runners to visualize their estimated finish distribution in real time."