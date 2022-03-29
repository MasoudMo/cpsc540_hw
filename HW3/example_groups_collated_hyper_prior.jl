using JLD
using SpecialFunctions
using Distributions
using Printf
data = load("cancerData.jld")
n = data["n"]
nTest = data["nTest"]

n_total_train = 0
n_one_train = 0

# collate samples
k = size(n, 1)
for i in 1:k
    n_total_train += n[i,2]
    n_one_train += n[i,1]
end
n_zero_train = n_total_train - n_one_train

# valid_values = []
# for x in range(start=0, stop=9, step=1)
#     for y in range(start=0, stop=0.9, step=0.1)
#         if x+y > 0
#             append!(valid_values, x+y)
#         end
#     end
# end

best_LL = -99999
best_a = -10
best_b = -10
best_m = -10
best_k = -10
for a in range(start=0.1, stop=10, step=0.1)
    for b in range(start=0.1, stop=10, step=0.1)
        m = a/(a+b)
        k = a+b
        LL = logbeta(n_one_train+a, n_zero_train+b) - logbeta(a,b) - 0.99*log(m) + 8.9*log(1-m) - 2*log(1+k)
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