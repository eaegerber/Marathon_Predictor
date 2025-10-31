# remove.packages(c("StanHeaders", "rstan"))
# if (file.exists(".RData")) file.remove(".RData")
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# example(stan_model, package = "rstan", run.dontrun = TRUE)
# install.packages("loo")

# getwd()
set.seed(2025)
library("rstan") # observe startup messages
library("loo")

features1 <- c("alpha", "total_pace")
features2 <- c("alpha", "total_pace", "curr_pace", "male")
features3 <- c("alpha", "total_pace", "curr_pace", "male", "age_t")

for (race in c("bos")) {
    print(race)
    train_name <- paste("processed_data/train_", race, ".csv", sep="")
    train_data <- read.csv(train_name)
    
    data1 <- list(N = nrow(train_data), K = length(features1), L = 8,
                  feats = train_data[features1], ll = train_data$lvl,
                  finish = train_data$finish)
    res_name1 <- paste("stan_results/model1/result_", race, ".csv", sep="")
    par_name1 <- paste("stan_results/model1/params_", race, ".csv", sep="")
    fit1 <- stan(file = 'marathon.stan', data = data1,
                 iter=800, chains=2, cores=2, seed=2025,
                 control = list(max_treedepth = 12))
    parameters1 <- as.data.frame(extract(fit1)[c("beta", "sigma", "lp__")])
    write.csv(parameters1, par_name1, row.names = TRUE)
    check_hmc_diagnostics(fit1)
    llk1 <- extract_log_lik(fit1, parameter_name = "log_lik", merge_chains = TRUE)
    loo1 <- loo(llk1)
    print(loo1)
    
    data2 <- list(N = nrow(train_data), K = length(features2), L = 8,
                  feats = train_data[features2], ll = train_data$lvl,
                  finish = train_data$finish)
    res_name2 <- paste("stan_results/model2/result_", race, ".csv", sep="")
    par_name2 <- paste("stan_results/model2/params_", race, ".csv", sep="")
    fit2 <- stan(file = 'marathon.stan', data = data2,
                 iter=800, chains=2, cores=2, seed=2025,
                 control = list(max_treedepth = 12))
    parameters2 <- as.data.frame(extract(fit2)[c("beta", "sigma", "lp__")])
    write.csv(parameters2, par_name2, row.names = TRUE)
    check_hmc_diagnostics(fit2)
    llk2 <- extract_log_lik(fit2, parameter_name = "log_lik", merge_chains = TRUE)
    loo2 <- loo(llk2)
    print(loo2)
    
    data3 <- list(N = nrow(train_data), K = length(features3), L = 8,
                  feats = train_data[features3], ll = train_data$lvl,
                  finish = train_data$finish)
    res_name3 <- paste("stan_results/model3/result_", race, ".csv", sep="")
    par_name3 <- paste("stan_results/model3/params_", race, ".csv", sep="")
    fit3 <- stan(file = 'marathon.stan', data = data3,
                 iter=800, chains=2, cores=2, seed=2025,
                 control = list(max_treedepth = 12))
    parameters3 <- as.data.frame(extract(fit3)[c("beta", "sigma", "lp__")])
    write.csv(parameters3, par_name3, row.names = TRUE)
    check_hmc_diagnostics(fit3)
    llk3 <- extract_log_lik(fit3, parameter_name = "log_lik", merge_chains = TRUE)
    loo3 <- loo(llk3)
    print(loo3)
    
    comp <- loo_compare(loo1, loo2, loo3)
    print(comp, simplify=FALSE)
    stan_rhat(fit3)
    traceplot(fit3, inc_warmup=TRUE)
    stan_ess(fit3)
}

# saveRDS(fit, file = "my_stan_fit.rds")
# print(names(extract(fit)))
# x = extract(fit)["finish_test"]
# summary(fit, pars=c('alpha', "beta", 'sigma'), probs=c(.25, .75))$summary
