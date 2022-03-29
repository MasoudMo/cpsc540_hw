using JLD, Printf, Statistics
using Distributions

# Load X and y variable
data = load("gaussNoise.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

X = X .- mean(X)
Xtest = Xtest .- mean(Xtest)

include("tda.jl")
model = tda(X,y,maximum(y))

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with TDA: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with TDA: %.3f\n",testError)