
# # Bayesian Linear Regression model

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from utils import int_to_str_time
# from utilsb import *

# import bambi as bmb
# np.random.seed(0)


# # def save_traces1(marks, X_train, sample_n=2000, draws=1000, tune=100, chains=1):
# #     traces = []
# #     for filename in marks:
# #         data = X_train[X_train["dist"] == filename].sample(sample_n)
# #         trace1 = bmb.Model("finish ~ total_pace + curr_pace", data).fit(
# #             draws=draws, tune=tune, discard_tuned_samples=True, chains=chains, progressbar=True
# #         )
# #         trace1.to_netcdf(f"traces/X{filename}.nc")
# #         traces.append(trace1)
# #         print(f"done: {filename}")
# #     return traces


# def _get_table_info(arr, dist):
#     median = int_to_str_time(np.percentile(arr, 50))
#     lower50, upper50 = int_to_str_time(np.percentile(arr, 25)), int_to_str_time(np.percentile(arr, 75))
#     lower80, upper80 = int_to_str_time(np.percentile(arr, 10)), int_to_str_time(np.percentile(arr, 90))
#     lower95, upper95 = int_to_str_time(np.percentile(arr, 2.5)), int_to_str_time(np.percentile(arr, 97.5))
#     range50, range80, range95 = f"{lower50}-{upper50}", f"{lower80}-{upper80}", f"{lower95}-{upper95}"
#     return f"After {dist}", median, range50, range80, range95


# def get_posterior_array(
#         train_data: pd.DataFrame,
#         test_data: pd.DataFrame,
#         trace,
# ):
#     model = bmb.Model("finish ~ total_pace + curr_pace", train_data)
#     pred_trace = model.predict(trace, data=test_data, inplace=False, kind='pps')
#     return pred_trace  # .posterior_predictive["Finish"]


# # def get_from_info(
# #     info: dict,
# #     train_data: pd.DataFrame,
# #     traces_map,
# #     name: str = "",
# #     actual=None,
# #     show: list = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
# # ):
# #     """Returns figure and table"""
# #     marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
# #     marks = [m for m in marks if m in info.keys()]
# #     shows = [m for m in marks if m in show]
# #     fig = plt.figure(figsize=(12, 10))
# #     table = []
# #     colors = plt.get_cmap()(np.linspace(0.2, 0.8, len(show)))
# #     for i, dist in enumerate(shows):
# #         test_data = pd.DataFrame([info[dist]], columns=["total_pace", "curr_pace"])
# #         test_data = time_to_pace(test_data, "Finish Net")
# #         trace = traces_map[f"X{dist}"]
# #         pred_trace = get_posterior_array(train_data, test_data, trace)
# #         arr1 = pred_trace.posterior_predictive["finish"]
# #         arr2 = np.array(arr1).flatten()
# #         arr = pd.Series(42195 / arr2 / 60)
# #         sns.kdeplot(arr, color=colors[i], label=dist)
# #         table.append(_get_table_info(arr, dist))

# #     x_labels = plt.xticks()[0]
# #     plt.xticks(x_labels, [int_to_str_time(t) for t in x_labels])
# #     plt.legend()
# #     plt.xlabel("Time (HH:MM)", fontsize=15)
# #     plt.ylabel("Probability", fontsize=15)
# #     plt.title(f"{name} Live Prediction", fontsize=17)
# #     if actual:
# #         plt.vlines(actual, 0, plt.yticks()[0].max(), linestyles="dashed", color="black", label="actual")
# #     print('done')
# #     return fig, pd.DataFrame(table, columns=["dist", "median", "range_50", "range_80", "range_95"])


# # def get_for_person(
# #     num, train_data, test_data, traces_map, save=False,
# #     show: list = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"], name=""
# # ):
# #     person = test_data[test_data["id"] == num] 
# #     actual = pace_to_time(person["finish"], "Finish Net").iloc[0]
# #     person[["total_pace", "curr_pace"]] = time_to_pace(person[["total_pace", "curr_pace"]], "Finish Net")
# #     person_dict = {row["dist"]: (row["total_pace"], row["curr_pace"]) for _, row in person.iterrows()}
# #     fig, table = get_from_info(person_dict, train_data=train_data, traces_map=traces_map,
# #                                name=name, actual=actual, show=show)
# #     if save:
# #         plt.savefig(f"plots/HalfPlot: {person['Name']}")
# #     return fig, table, actual


# def bayes_reg(train, test, trace_map, dist="X5K", quartile=50):
#     model = bmb.Model("finish ~ total_pace + curr_pace", train)
#     pred_trace = model.predict(trace_map[dist], data=test, inplace=False, kind='pps')   
#     y_pred_bayes = np.percentile(pred_trace.posterior_predictive["finish"][0], q=quartile, axis=0)
#     y_pred_bayes = np.array(42195 / y_pred_bayes / 60)
#     return y_pred_bayes

# def bayes_predictions(train, test, trace_map, dist="X5K"):
#     model = bmb.Model("finish ~ total_pace + curr_pace", train)
#     pred_trace = model.predict(trace_map[dist], data=test, inplace=False, kind='pps')   
#     y_pred_bayes = np.array(pred_trace.posterior_predictive["finish"][0])
#     return y_pred_bayes

# def get_bayes(bayes_preds, quantiles):
#     preds = 42195 / np.percentile(bayes_preds, q=quantiles, axis=0) / 60
#     return preds

# if __name__ == "__main__":
#     nucr_filename = "processed_data/nucr_runners.csv"
#     nucr = pd.read_csv(nucr_filename)
#     names = nucr["Name"]
#     X_train, X_test, NUCR_test, trace_map = get_data()
#     marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
#     mapping = {name: i for i, name in enumerate(names)}
#     # for name in names:
#     #     i = mapping[name]
#         # fig, table, actual = get_for_person(i, train_data=X_train, test_data=NUCR_test, traces_map=trace_map, save=False, show=marks)
#         # plt.show()