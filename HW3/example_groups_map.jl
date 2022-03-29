using JLD
data = load("cancerData.jld")
n = data["n"]
nTest = data["nTest"]

# Compute number of groups (here, the number of training and testing groups is the same)
k = size(n,1)

# Function to compute NLL when we have a theta for each group
NLLs(theta,n) =
begin
    LL = 0
    for j in 1:k
        LL -= n[j,1]*log(theta[j]) + (n[j,2]-n[j,1])*log(1-theta[j])
    end
    return LL
end

# Fit MAP for each training group
a = 1
b = 731
MAPs = (n[:,1].+a.-1)./(n[:,2].+a.+b.-2)

# Show training and test NLL
@show NLLs(MAPs,nTest)
