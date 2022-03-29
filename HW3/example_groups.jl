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

# Function to compute NLL when we have a theta for each group
LLs(theta,n) =
begin
    LL = 1
    for j in 1:k
        LL *= theta[j]^n[j,1] * (1-theta[j])^(n[j,2]-n[j,1])
    end
    return LL
end

# Fit MLE for each training group
MLEs = n[:,1]./n[:,2]
@show LLs(MLEs,nTest)


# Show test NLL if all theta=0.5
theta = 0.5*ones(k)
@show NLLs(theta,nTest)


# Show training and test NLL for MLE
@show NLLs(MLEs,nTest)
