
import stan
import numpy as np
import pandas as pd

np.random.seed(2024)
with open("marathon.stan", "r") as f:
    schools_code = f.read()

train_df = pd.read_csv("train.csv")#[:50]
print(len(train_df))

def single_model(stan_code, data_dict, outpath="stan_results/ps_result.csv"):
    """Fit a single stan model, incorporating all dists"""
    posterior = stan.build(stan_code, data=data_dict, random_seed=2)
    fit = posterior.sample(num_chains=4, num_samples=1000)
    df = fit.to_frame()
    df.to_csv(outpath)
    return # posterior, fit

def separate_models(stan_code, data, feats_sep, outdir="stan_results"):
    """Fit an inividual stan model for each dist"""
    marks = ["5K", "10K", "15K", "20K", "25K", "30K", "35K", "40K"]
    for mk in marks:
        print("loading: ", mk)
        df = data[data["dist"] == mk]
        schools_data = {"feats": df[feats_sep].values, "finish": df["finish"].values, "N": len(df), "K": len(feats_sep)}
        single_model(schools_code, schools_data, outpath=f"{outdir}/ps_result{mk}.csv")
    return

print('start')
outdir = "stan_results/bayes1"
feats = ["total_pace", "prop"]
schools_data = {"feats": train_df[feats].values, "finish": train_df["finish"].values, "N": len(train_df), "K": len(feats)}
single_model(stan_code=schools_code, data_dict=schools_data, outpath=f"{outdir}/ps_result.csv")
separate_models(stan_code=schools_code, data=train_df, feats_sep=feats[:-1], outdir=outdir)

outdir = "stan_results/bayes2"
feats = ["total_pace", "curr_pace", "prop"]
schools_data = {"feats": train_df[feats].values, "finish": train_df["finish"].values, "N": len(train_df), "K": len(feats)}
single_model(stan_code=schools_code, data_dict=schools_data, outpath=f"{outdir}/ps_result.csv")
separate_models(stan_code=schools_code, data=train_df, feats_sep=feats[:-1], outdir=outdir)
print('end')


# if __name__ == "__main__":
#     print('hi')
#     a(schools_code, schools_data)
#     print('bye')

