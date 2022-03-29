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

last_step = 10
num_sims = 10000

# Generate 10000 samples of length 10
samples = sampleAncestral(p1, pt, num_sims, last_step)

# Reject samples that do not match the x10 = 6 conditional
samples = samples[samples[:, last_step] .== 6, :]
accepted_samples = size(samples, 1)
@printf("Number of accepted samples: %d \n", accepted_samples)

# Extract the 5th column
state_of_interest = samples[:, 5]

# Find the marginal probabilities
for i in 1:k
    n_i = sum(state_of_interest .== i)
    @printf("MC Estimated probability of %d th category is %f \n", i, n_i/num_sims)
end

# Exact forward conditional with fb algortihm
prob = fb_algorithm(p1, pt, k, last_step, 6, 5)
for i in 1:k
    @printf("FB alogrithm probability of %d th category is %f \n", i, prob[i])
end
