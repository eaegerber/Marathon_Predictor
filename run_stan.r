# remove.packages(c("StanHeaders", "rstan"))
# if (file.exists(".RData")) file.remove(".RData")
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# example(stan_model, package = "rstan", run.dontrun = TRUE)

# getwd()
library("rstan") # observe startup messages
train_data <- read.csv("processed_data/train.csv")
test_data <- read.csv("processed_data/test.csv")
features <- c("total_pace", "curr_pace", "prop") #, "propxcurr", "male", "age", "malexage")

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

write.csv(predictions,"stan_results/result2.csv", row.names = TRUE)
write.csv(parameters,"stan_results/params2.csv", row.names = TRUE)

d2 <- apply(extract(fit)$finish_test, 2, sd)

print(names(extract(fit)))
summary(fit, pars=c('alpha', "beta", 'sigma'), probs=c(.25, .75))$summary

# d6 <- as.data.frame(test_data$finish)
