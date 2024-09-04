
# Quantile Linear Regression model

import numpy as np
import statsmodels.formula.api as smf
np.random.seed(0)


def get_quant(train, test, formula: str = "finish ~ total_pace + curr_pace",  quantile=50):
    """Quantile should be [0, 100]"""
    quantile = 1 - (quantile / 100)
    mod = smf.quantreg(formula, train)
    res_med = mod.fit(q=quantile)
    y_pred_quar = res_med.predict(test)
    y_pred_quar = (42195 / 60) / np.array(y_pred_quar)
    return y_pred_quar


def get_quants(train, test, formula: str = "finish ~ total_pace + curr_pace", quantiles=[50]):
    """Quantiles should be [0, 100]"""
    mod = smf.quantreg(formula, train)
    preds = []
    for q in quantiles:
        quantile = 1 - (q / 100)
        res_med = mod.fit(q=quantile)
        pred = res_med.predict(test)
        preds.append(pred)

    y_pred_quar = (42195 / 60) / np.array(preds)
    return y_pred_quar

# if __name__ == "__main__":
    # X_train, X_test, X2_test, traces_map = get_data()