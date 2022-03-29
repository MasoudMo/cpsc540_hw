# Load X and y variable
using JLD
using(PyPlot)
data = load("nonLinear.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Compute number of training examples and number of features
(n,d) = size(X)

# Fit least squares model
include("leastSquares.jl")
leastSquaresEmpiricalBaysis(X, y, 5, 8, 8)
