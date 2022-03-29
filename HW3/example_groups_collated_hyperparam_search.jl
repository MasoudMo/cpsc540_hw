using JLD
using SpecialFunctions
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

# Fit MLE for each training group
MLE = n_one_train/n_total_train

best_k = -999999
best_ll = -999999
for k in 0:10:100000
    ll = logbeta(MLE*k+n_one_train, k*(1-MLE)+n_zero_train) - logbeta(MLE*k, k*(1-MLE))
    if ll > best_ll
        best_k = k
        best_ll = ll
    end
end

@printf("The best K is %f \n",best_k)
@printf("The best LL is %f",best_ll)
