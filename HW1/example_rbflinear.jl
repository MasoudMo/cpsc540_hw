# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])

# Compute number of training examples and number of features
(n,d) = size(X)

# Fit least squares model
include("leastSquares.jl")
include("leastSquaresRBFL2.jl")
linmodel = leastSquares(X,y)

yprime = linmodel.predict(X)
yprime = y .- yprime

model = leastSquaresRBFL2(X, yprime, lambda=1, sigma=1);

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
y1 = linmodel.predict(Xhat)
y2 = model.predict(Xhat)
yhat = y1 .+ y2;
plot(Xhat[:],yhat,"r")
ylim((0,2.5))
display(gcf())