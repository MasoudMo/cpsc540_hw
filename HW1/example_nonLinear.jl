# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])

# Compute number of training examples and number of features
(n,d) = size(X)

# Fit least squares model
include("leastSquares.jl")
model = leastSquares(X,y)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.1:maximum()
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat[:],yhat,"r")
ylim((0,2.5))
display(gcf())