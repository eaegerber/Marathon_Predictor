
# Bayesian Linear Regression model

import numpy as np
from utils import *
import statsmodels.formula.api as smf
np.random.seed(0)


def quart_reg(train, test, dist="X5K", quartile=0.5):
    mod = smf.quantreg("finish ~ total_pace + curr_pace", train)
    res_med = mod.fit(q=quartile)
    y_pred_quar = res_med.predict(test)

    y_pred_quar = np.array(42195 / y_pred_quar / 60)
    return y_pred_quar


def get_quant(train, test, quantiles, dist="X5K"):
    mod = smf.quantreg("finish ~ total_pace + curr_pace", train)
    preds = []
    for q in quantiles:
        res_med = mod.fit(q=q)
        pred = res_med.predict(test)
        preds.append(pred)

    y_pred_quar = 42195 / np.array(preds) / 60
    return y_pred_quar

# if __name__ == "__main__":
    # X_train, X_test, X2_test, traces_map = get_data()