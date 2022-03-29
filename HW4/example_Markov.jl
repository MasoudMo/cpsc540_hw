# Load X and y variable
using JLD, Printf
include("markov_funcs.jl")

# Load initial probabilities and transition probabilities of Markov chain
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])

# Set 'k' as number of states
k = length(p1)

# Confirm that initial probabilities sum up to 1
@show sum(p1)

# Confirm that transition probabilities sum up to 1, starting from each state
@show sum(pt,dims=2)

last_step = 50
num_sims = 10000

# Generate 10000 samples of length 50
samples = sampleAncestral(p1, pt, num_sims, last_step)

# Extract the last column
state_of_interest = samples[:, last_step]

# Find the marginal probabilities
for i in 1:k
    n_i = sum(state_of_interest .== i)
    @printf("MC Estimated Marginal probability of %d th category is %f \n", i, n_i/num_sims)
end

last_step = 50

# Use CK to find exact marginals
marginals = marginalCK(p1, pt, last_step)

# Print the marginal probabilities
for i in 1:k
    @printf("CK Marginal probability of %d th category is %f \n", i, marginals[i])
end

# Find c with highest probability at each step
for last_step in 1:100
    marginals = marginalCK(p1, pt, last_step)
    @printf("c with max marginal probability at step %d is %d \n", last_step, argmax(marginals))
end

# Conditional CK
p1 = zeros(k)
p1[3] = 1
# Use CK to find exact marginals
marginals = marginalCK(p1, pt, 50)
# Print the marginal probabilities
for i in 1:k
    @printf("CK conditional marginal probability of %d th category is %f \n", i, marginals[i])
end