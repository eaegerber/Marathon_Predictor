# remove.packages(c("StanHeaders", "rstan"))
# if (file.exists(".RData")) file.remove(".RData")
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# example(stan_model, package = "rstan", run.dontrun = TRUE)

# getwd()
library("rstan") # observe startup messages

features1 <- c("alpha", "total_pace")
features2 <- c("alpha", "total_pace", "curr_pace")
features3 <- c("alpha", "total_pace", "curr_pace", "male", "age")
features4 <- c("alpha", "total_pace", "curr_pace", "male", "age", "malexage")
feat_list = list(features1, features2, features3, features4)

for (num in 1:1) {
  for (race in c("bos", "nyc", "chi")) {
    print(race)
    print(num)
    features <- feat_list[[num]]
    print(features)
    train_name <- paste("processed_data/train_", race, ".csv", sep="")
    # test_name <- paste("processed_data/test_", race, ".csv", sep="")
    res_name <- paste("stan_results/model", num, "/result_", race, ".csv", sep="")
    par_name <- paste("stan_results/model", num, "/params_", race, ".csv", sep="")
  
    train_data <- read.csv(train_name)
    # test_data <- read.csv(test_name)
    s_dat <- list(N = nrow(train_data),
                  K = length(features),
                  L = 8,
                  feats = train_data[features],
                  ll = train_data$lvl,
                  finish = train_data$finish)#,
                  # N_test = nrow(test_data),
                  # feats_test = test_data[features],
                  # ll_test = test_data$lvl)
  
    fit <- stan(file = 'marathon.stan', data = s_dat, cores=4)
    # predictions <- colMeans(extract(fit)$finish_test)
    parameters <- as.data.frame(extract(fit)[c("beta", "sigma", "lp__")])
    # write.csv(predictions, res_name, row.names = TRUE)
    write.csv(parameters,par_name, row.names = TRUE)
  }
}


# d2 <- apply(extract(fit)$finish_test, 2, sd)
# print(names(extract(fit)))
# x = extract(fit)["finish_test"]
# summary(fit, pars=c('alpha', "beta", 'sigma'), probs=c(.25, .75))$summary
# d6 <- as.data.frame(test_data$finish)
