# remove.packages(c("StanHeaders", "rstan"))
# if (file.exists(".RData")) file.remove(".RData")
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# example(stan_model, package = "rstan", run.dontrun = TRUE)

# getwd()
library("rstan") # observe startup messages

train_name <- "processed_data/train_chi.csv"
# "processed_data/train_bos.csv"
test_name <- "processed_data/test_chi.csv"
# "processed_data/test_bos.csv"
result_name <- "stan_results/result_chi2.csv"
# "stan_results/result_bos2.csv"
param_name <- "stan_results/params_chi2.csv"
# "stan_results/params_bos2.csv"

#features <- c("total_pace", "prop") 
features <- c("total_pace", "curr_pace", "prop") #, "propxcurr", "male", "age", "malexage")

train_data <- read.csv(train_name)
test_data <- read.csv(test_name)
schools_dat <- list(N = nrow(train_data),
                    K = length(features),
                    feats = train_data[features],
                    propleft = train_data$propleft,
                    finish = train_data$finish,
                    N_test = nrow(test_data),
                    feats_test = test_data[features],
                    propleft_test = test_data$propleft)

fit <- stan(file = 'marathon.stan', data = schools_dat)

predictions <- colMeans(extract(fit)$finish_test)
parameters <- as.data.frame(extract(fit)[c("alpha", "beta", "sigma", "lp__")])

write.csv(predictions, result_name, row.names = TRUE)
write.csv(parameters,param_name, row.names = TRUE)

d2 <- apply(extract(fit)$finish_test, 2, sd)

print(names(extract(fit)))
summary(fit, pars=c('alpha', "beta", 'sigma'), probs=c(.25, .75))$summary

# d6 <- as.data.frame(test_data$finish)
