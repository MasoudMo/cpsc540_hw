using Printf

# Load X and y variable
using JLD
using PyPlot
include("findMin.jl")
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])
# add bias
(n,d) = size(X)
X = [ones(n,1) X]
(n,d) = size(X)


# Choose network structure and randomly initialize weights
include("newNeuralNet.jl")
nHidden = [15, 5, 3]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

maxIter = 1000000
stepSize = 1e-4
# for t in 1:maxIter

# 	# The stochastic gradient update:
# 	i = rand(1:n, 100)
# 	(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
# 	global w = w - stepSize*g

# 	# Every few iterations, plot the data/model:
# 	if (mod(t-1,round(maxIter/50)) == 0)
# 		@printf("Training iteration = %d\n",t-1)
# 		figure(1)
# 		clf()
# 		xVals = -10:.05:10
# 		Xhat = zeros(length(xVals),1)
# 		Xhat[:] .= xVals
# 		Xhat = [ones(size(Xhat)[1],1) Xhat]
# 		yhat = NeuralNet_predict(w,Xhat,nHidden)
# 		plot(X[:, 2:end],y,".")
# 		plot(Xhat[:, 2:end],yhat,"g-")
# 		sleep(.1)
# 	end
# end


objective(w) = NeuralNet_backprop(w,X,y,nHidden)
w = findMin(objective,w,maxIter=maxIter)

figure(1)
clf()
xVals = -10:.05:10
Xhat = zeros(length(xVals),1)
Xhat[:] .= xVals
Xhat = [ones(size(Xhat)[1],1) Xhat]
yhat = NeuralNet_predict(w,Xhat,nHidden)
plot(X[:, 2:end],y,".")
plot(Xhat[:, 2:end],yhat,"g-")
sleep(.1)


display(gcf())
