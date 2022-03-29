using JLD
using SpecialFunctions
using Distributions
using Printf
data = load("cancerData.jld")
n = data["n"]
nTest = data["nTest"]

num_groups = size(n, 1)

valid_values = []
for x in range(start=0, stop=1000, step=1)
    for y in range(start=0, stop=0.9, step=0.1)
        if x+y > 0
            append!(valid_values, x+y)
        end
    end
end

best_LL = -99999
best_a = -10
best_b = -10
best_m = -10
best_k = -10
for a in range(1, 1000)
    for b in range(1, 1000)
        m = a/(a+b)
        k = a+b
        LL = 0
        for group in 1:num_groups
            n_one = n[group,1]
            n_zero = n[group,2] - n[group,1]
            LL += logbeta(n_one+a, n_zero+b) - logbeta(a,b) 
        end
        LL += -0.99*log(m) + 8.9*log(1-m) - 2*log(1+k)
        if LL > best_LL
            best_LL = LL
            best_b = b
            best_a = a
            best_m = m
            best_k = k
        end
    end
end


@printf("Best LL is %f \n", best_LL)
@printf("Best a is %f \n", best_a)
@printf("Best b is %f \n", best_b)
@printf("Best m is %f \n", best_m)
@printf("Best k is %f \n", best_k)