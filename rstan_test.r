# remove.packages(c("StanHeaders", "rstan"))
# if (file.exists(".RData")) file.remove(".RData")
# install.packages("rstan", repos = "https://cloud.r-project.org/", dependencies = TRUE)
# example(stan_model, package = "rstan", run.dontrun = TRUE)
library("rstan") # observe startup messages
train_data <- read.csv("/Users/brandononyejekwe/Documents/Personal_Projects/Marathon_Predictor/processed_data/train.csv")
features <- c("total_pace", "curr_pace", "prop") #, "propxcurr", "male", "age", "malexage")

schools_dat <- list(N = nrow(train_data),
                    K = length(features),
                    feats = train_data[features],
                    propleft = train_data$propleft,
                    finish = train_data$finish)

fit <- stan(file = '/Users/brandononyejekwe/Documents/Personal_Projects/Marathon_Predictor/marathon.stan', data = schools_dat)
d <- as.data.frame(fit)
write.csv(d,"/Users/brandononyejekwe/Documents/Personal_Projects/Marathon_Predictor/stan_results/rs_result2dw.csv", row.names = TRUE)
# getwd()



