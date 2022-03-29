using JLD
using SpecialFunctions
data = load("cancerData.jld")
n = data["n"]
nTest = data["nTest"]

n_total_train = 0
n_pos_train = 0
n_total_test = 0
n_pos_test = 0

# collate samples
for i in 1:size(n, 1)
    n_total_train += n[i,2]
    n_pos_train += n[i,1]
end
for i in 1:size(nTest, 1)
    n_total_test += nTest[i,2]
    n_pos_test += nTest[i,1]
end

# Function to compute NLL
NLLs(theta,n_pos,n_total) =
begin
    LL = -(n_pos*log(theta) + (n_total-n_pos)*log(1-theta))
end

# Function to compute NLPP
NLPP(n_pos_test, n_total_test, n_pos_train, n_total_train, a, b) =
begin
    LL = -log(beta(a+n_pos_train+n_pos_test, b + (n_total_train - n_pos_train) + (n_total_test - n_pos_test)))
    LL += log(beta(a+n_pos_train, b+(n_total_train - n_pos_train)))
end

# Show test NLL if all theta=0.5
theta = 0.5
@show NLLs(theta,n_pos_test, n_total_test)

# Fit MLE for each training group
MLE = n_pos_train/n_total_train

# Show training and test NLL for MLE
@show NLLs(MLE,n_pos_test, n_total_test)

# Fit MAP for each training group
a = 2
b = 2
MAP = (n_pos_train+a-1)/(n_total_train+a+b-2)

# Show training and test NLL for MLE
@show NLLs(MAP,n_pos_test, n_total_test)

# Show training and test NLL for MLE
@show NLPP(n_pos_test, n_total_test, n_pos_train, n_total_train, a, b)

