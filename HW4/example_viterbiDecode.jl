using JLD, Printf
include("viterbiDecode.jl")

# Load initial probabilities and transition probabilities of Markov chain
data = load("gradChain.jld")
(p1,pt) = (data["p1"],data["pt"])

# Set 'k' as number of states
k = length(p1)

steps = 50
decoded_chain = viterbiDecode(p1, pt, k, steps)
@printf("The decoded chain for d=50 is: \n")
print(decoded_chain)

steps = 100
decoded_chain = viterbiDecode(p1, pt, k, steps)
@printf("\n The decoded chain for d=100 is: \n")
print(decoded_chain)