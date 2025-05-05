data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] feats;
  vector[N] finish;
  vector[N] propleft;
}
parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> sigma;
}
model {
  // priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);
  sigma ~ cauchy(0, 2.5); 

  // model
  finish ~ normal(feats * beta + alpha, sigma * propleft);
}