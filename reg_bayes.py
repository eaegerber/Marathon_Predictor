
# # Bayesian Linear Regression model

# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from utils import int_to_str_time

# import bambi as bmb
# np.random.seed(0)

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