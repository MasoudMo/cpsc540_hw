using JLD
using SpecialFunctions
data = load("cancerData.jld")
n = data["n"]
nTest = data["nTest"]

# Compute number of groups (here, the number of training and testing groups is the same)
k = size(n,1)

# Function to compute NLPP when we have a theta for each group
NLPP(n_test, n_train, a, b) =
begin
    LL = 0
    for j in 1:k
        LL -= log(beta(a+n_train[j, 1]+n_test[j, 1], b + (n_train[j,2] - n_train[j, 1]) + (n_test[j, 2] - n_test[j, 1])))
        LL += log(beta(a+n_train[j, 1], b+(n_train[j, 2] - n_train[j, 1])))
    end
    return LL
end

# Show training and test NLL for MLE
@show NLPP(nTest, n, 1, 731)
