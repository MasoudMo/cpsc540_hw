# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Fit Naive Bayes model
X[X.>0.5] .= 2
X[X.<0.5] .= 1
X = Int64.(X)
Xtest[Xtest.>0.5] .= 2
Xtest[Xtest.<0.5] .= 1
Xtest = Int64.(Xtest)
include("NaiveBayes.jl")
model = NaiveBayes(X,y)

## Compute error on test data
yhat = model.predict(Xtest)
err = sum(yhat .!= ytest)/size(Xtest,1)
@printf("Error rate = %.2f\n",err)
