# Marathon_Predictor
Bayesian model for in-race marathon finish time predictions.

## Quantifying Uncertainty in Live Marathon Finish Time Predictions
In the middle of a marathon, expected finish times are traditionally estimated by naively extrapolating the average
pace covered so far, assuming it will be held constant for the rest of the race. Making predictions like this causes
two issues: (1) the estimates do not consider context occurring within the race that can determine if a runner is
likely to finish much slower or faster than expected, and (2) the prediction is a single point estimate with no 
information about uncertainty. A Bayesian inference model addresses both concerns by using the runner's previous 
splits in the race to generate a probability distribution of possible finish times, using conditional probability 
and empirical likelihood estimates. In this project, a Bayesian model is evaluated in comparison to the traditional
estimate method.

Data:
	We scraped the data for this project from the Boston Athletic Association website. It contains a runner's name,
age, gender, the intermediate splits of their race (5K, 10K, 15K, 20K, HALF, 25K, 30K, 35K, 40K) as well as their
finish time. The splits and finish times are recorded in seconds. The data includes every finishing runner from each
Boston Marathon from 2009-2023, resulting in 312805 rows of data. The data was partitioned into two groups: a 
training set (286777 runners from 2009-2022) and a test set (26028 runners from 2023). 

Model:
    The model utilizes Bayes theorem to iteratively update the posterior finish time distribution according to the
following general equation: 

`P(finish | current_split, previous_splits) ∝ P(current_split | finish, previous_splits) * P(finish | previous_splits)`. 

In this equation,
the posterior distribution is the normalized product of the likelihood distribution and the prior distribution. 


-------
Notes:


### `POSTERIOR ∝ LIKELIHOOD * PRIOR`

•	`P(finish | 5K) ∝ P(5K | finish) * P(finish)`

•	`P(finish | 5K, 10K) ∝ P(10K | finish, 5K) * P(finish|5K)`

•	`P(finish | 5K, 10K, 15K) ∝ P(15K | finish, 10K) * P(finish | 5K, 10K)`

•	`P(finish | 5K, 10K, 15K, 20K) ∝ P(20K | finish, 15K) * P(finish | 5K, 10K, 15K)`



Assumption: P(10K | finish, 5K, 10K) simplifies to P(15K|finish)  - BDA pg. 11
•	…

Initial prior distribution: density of all finish times in dataset (probably should filter using information about runner like age, sex, experience, etc.)


