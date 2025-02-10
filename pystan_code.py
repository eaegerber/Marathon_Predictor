import stan
import numpy as np
import pandas as pd

np.random.seed(2024)
with open("marathon.stan", "r") as f:
    schools_code = f.read()

train_df = pd.read_csv("train.csv")#[:50]
print(len(train_df))

feats = ["total_pace", "curr_pace", "prop"]
schools_data = {"feats": train_df[feats].values, "finish": train_df["finish"].values, "N": len(train_df), "K": len(feats)}


def single_model(stan_code, data_dict, outpath="stan_results/ps_result.csv"):
    posterior = stan.build(stan_code, data=data_dict, random_seed=2)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    df = fit.to_frame()
    df.to_csv(outpath)
    return # posterior, fit

def separate_models(stan_code, data):
    feats = ["total_pace", "curr_pace"]

    marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    for mk in marks:
        print("loading: ", mk)
        df = data[data["dist"] == mk]
        schools_data = {"feats": df[feats].values, "finish": df["finish"].values, "N": len(df), "K": len(feats)}
        single_model(schools_code, schools_data, outpath=f"stan_results/ps_result{mk}.csv")
    return

print('start')
single_model(schools_code, schools_data)
separate_models(schools_code, train_df)

# posterior, fit = single_model(schools_code, schools_data)
print('end')

# import arviz as az
# idata = az.from_pystan(posterior=fit, posterior_model=posterior)

# print(idata)
# print(type(idata))
# print(type(fit))
# print(type(posterior))
# print(idata.posterior)
# print('a')
# print(idata.sample_stats)
# print(';done')
# idata.to_netcdf('save_idata.nc')


# if __name__ == "__main__":
#     print('hi')
#     a(schools_code, schools_data)
#     print('bye')

