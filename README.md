# Marathon_Predictor
Bayesian model for in-race marathon finish time predictions.

Code for the paper [Quantifying Uncertainty in Live Marathon Finish Time Predictions](https://github.com/bonyejekwe/Marathon_Predictor/blob/main/paper.pdf)

**Abstract:** In the middle of a marathon, a runner’s expected finish time is commonly estimated by extrapolating the average pace covered so far, assuming it to be constant for the rest of the race. These predictions have two key issues: the 
estimates do not consider the in-race context that can determine if a runner is likely to finish faster or slower than 
expected, and the prediction is a single point estimate with no information about uncertainty. We implement two approaches 
to address these issues: Bayesian linear regression and quantile regression. Both methods incorporate information from all 
splits in the race and allow us to quantify uncertainty around the predicted finish times. We utilized 15 years of Boston 
Marathon data (312,805 runners total) to evaluate and compare both approaches. Finally, we developed an app for runners to 
visualize their estimated finish distribution in real time.

To set up and run project locally, follow the steps in `analysis.ipynb`.