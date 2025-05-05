# remove.packages(c("StanHeaders", "rstan"))
# if (file.exists(".RData")) file.remove(".RData")
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# example(stan_model, package = "rstan", run.dontrun = TRUE)
library("rstan") # observe startup messages
# schools_dat <- list(J = 8, 
#                     y = c(28,  8, -3,  7, -1,  1, 18, 12),
#                     sigma = c(15, 10, 16, 11,  9, 11, 10, 18))

train_data <- read.csv("/Users/brandononyejekwe/Documents/Personal_Projects/Marathon_Predictor/train.csv")
features <- c("total_pace", "curr_pace", "prop")

schools_dat <- list(N = nrow(train_data),
                    K = length(features),
                    feats = train_data[features],
                    propleft = train_data$propleft,
                    finish = train_data$finish)


typeof(train_data["finish"])
train_data["finish"]
fit <- stan(file = '/Users/brandononyejekwe/Documents/Personal_Projects/Marathon_Predictor/marathon.stan', data = schools_dat)
d <- as.data.frame(fit)
typeof(d)
write.csv(d,"/Users/brandononyejekwe/Documents/Personal_Projects/Marathon_Predictor/stan_results/bayes2/rs_result.csv", row.names = TRUE)
getwd()



