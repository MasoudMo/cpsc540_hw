using JLD
using Printf

include("sampleBernoulli.jl")

t = 15;
theta = 0.12;
samples = sampleBernoulli(t, theta);

@printf("The %.d samples generated from Bernoulli with theta %.2f: \n", t, theta)
print(samples)