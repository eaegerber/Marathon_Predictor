
import faicons as fa
from shiny import reactive
from shiny.express import input, render, ui

import pandas as pd
from utils import int_to_str_time
from race_class import RaceSplits, get_from_info, get_race_for_person

nucr_filename = "processed_data/nucr_runners.csv"
nucr = pd.read_csv(nucr_filename)
names = nucr["Name"]
marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
race_splits = RaceSplits()
ui.page_opts(title="Quantifying Uncertainty in Live Marathon Finish Time Predictions", fillable=True)

with ui.sidebar(id="sidebar", open="open"):
    ui.input_radio_buttons("radio", "Input Type",  {"0": "Total Time", "1": "Last Split"}, inline=True)  
    ui.input_radio_buttons("radio2", "Marathon Race",  {"0": "Boston", "1": "New York", "2": "Chicago"}, inline=True)  
    ui.input_select("select", label="Choose a distance", choices=marks, selected="5K")
    ui.input_text("runner_split1", value="", label="Enter time here:", placeholder="MM:SS")
    ui.input_action_button("bttn", "Add Time", class_="btn-success")
    ui.input_action_button("reset", "Reset", class_="btn-success")

    with ui.card(full_screen=True):
        @render.text
        @reactive.event(input.bttn, input.reset)       
        def textSidebar():
            if input.bttn() > race_splits.bttn_count:
                split = True if input.radio() == "1" else False
                race_splits.add_pace(dist=input.select(), time=input.runner_split1(), split=split)
                race_splits.bttn_count += 1
                return f"Updated {input.select()}"
            else:
                race_splits.reset_race()
                return "Reset"

with ui.nav_panel("My Plot"):
    with ui.layout_columns(fill=False):
        with ui.card(full_screen=True):
            "View your finish time estimates in real time! Sequentially input your race splits (formatted MM:SS or HH:MM:SS) in the text box below, click go, and view your live prediction at that stage of the race. Input your times in 5km increments [5K, 10K, ..., 40K].  NOTE: if the app errors, refresh the page and try again."

    with ui.layout_columns(fill=False):
        with ui.card(full_screen=True):
            ui.card_header("Inputted Times")
            @render.data_frame
            @reactive.event(input.bttn, input.reset)
            def tableTimes():
                df = race_splits.stored_times_table()
                print('tbl')
                print(df)
                return df

    with ui.layout_columns(col_widths=[6, 6, 12]):

        @reactive.calc
        def getInfoInput():
            print(race_splits.get_stored_paces())
            if input.runner_split1():
                race_splits.city = {"0": "bos", "1": "nyc", "2": "chi"}[input.radio2()]
                fig, table = get_from_info(race_splits, show=marks)
                return fig, table
            else:
                return None, None
                                     
        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Table")

            with ui.popover(title="Filter credible intervals", placement="top"):        
                fa.icon_svg("gear")
                ui.input_checkbox_group("intervals1", "Credible Intervals", choices=["range_50", "range_80", "range_95"], 
                                        selected=["range_50", "range_95"], inline=True)
            
            @render.data_frame
            @reactive.event(input.bttn, input.reset)
            def tableInput():
                info_table = getInfoInput()[1]
                return info_table[["dist", "median"] + list(input.intervals1())]

        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Plot")

            @render.plot
            @reactive.event(input.bttn, input.reset)
            def histInput():
                return getInfoInput()[0]


with ui.nav_panel("NUCR Plots"):
    "View the plots of NUCR runners! The motivation behind this project involves the Northeastern Club Running team, which has dozens of Boston Marathon qualifiers every year. Here are some select NUCR runners that ran in the 2023 Boston Marathon race. The vertical dotted black line in the plot shows their actual finish time."
    ui.input_selectize("runner_name0", "Select Runner", choices=list(names))
    ui.input_checkbox_group("splits_list0", "Select Splits", choices=marks, selected=marks, inline=True)

    with ui.layout_columns(col_widths=[6, 6, 12]):

        @reactive.calc
        def getInfoNUCR():
            mapping = {name: i for i, name in enumerate(names)}
            name = input.runner_name0()
            i = mapping[name]

            race, actual = get_race_for_person(i, nucr)
            race_splits.city = {"0": "bos", "1": "nyc", "2": "chi"}[input.radio2()]
            fig, table = get_from_info(race, show=input.splits_list0(), actual=actual)
            return fig, table, actual
                                     
        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Table")

            @render.text        
            def textNUCR():     
                actual_time = getInfoNUCR()[2]
                return f"Actual finish time: {int_to_str_time(actual_time)}" #.txt()
            
            with ui.popover(title="Filter credible intervals", placement="top"):        
                fa.icon_svg("gear")
                ui.input_checkbox_group("intervals0", "Credible Intervals", choices=["range_50", "range_80", "range_95"], 
                                        selected=["range_50", "range_95"], inline=True)
            
            @render.data_frame
            def tableNUCR():
                info_table = getInfoNUCR()[1]
                return info_table[["dist", "median"] + list(input.intervals0())]

        with ui.card(full_screen=True):
            ui.card_header("Live Prediction: Plot")

            @render.plot
            def histNUCR():
                return getInfoNUCR()[0]
            

with ui.nav_panel("Project Info"):
    "Quantifying Uncertainty in Marathon Finish Time Predictions: In the middle of a marathon, a runner’s expected finish time is commonly estimated by extrapolating the average pace covered so far, assuming it to be constant for the rest of the race. These predictions have two key issues: the estimates do not consider the in-race context that can determine if a runner is likely to finish faster or slower than expected, and the prediction is a single point estimate with no information about uncertainty. We implement two approaches to address these issues: Bayesian linear regression and quantile regression. Both methods incorporate information from all splits in the race and allow us to quantify uncertainty around the predicted finish times. We utilized 15 years of Boston Marathon data (312,805 runners total) to evaluate and compare both approaches. Finally, we developed an app for runners to visualize their estimated finish distribution in real time."