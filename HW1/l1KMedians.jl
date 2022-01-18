using Printf
using Statistics
using Random
include("misc.jl")
include("clustering2Dplot.jl")
include("l1KMediansError.jl")

mutable struct PartitionModel
	predict # Function for clustering new points
	y # Cluster assignments
	W # Prototype points
end

function l1KMedians(X,k;doPlot=false)
	# K-medians clustering with L1 Norm
	
	(n,d) = size(X)
	
	# Choos random points to initialize medians
	W = zeros(k,d)
	perm = randperm(n)
	for c = 1:k
		W[c,:] = X[perm[c],:]
	end
	
	# Initialize cluster assignment vector
	y = zeros(Int64, n)
	changes = n
	while changes != 0
	
		# Compute L1 distance between each point and each median
		D = l1Distance(X,W)
	
		# Degenerate clusters will distance NaN, change to Inf
		# (since Julia thinks NaN is smaller than all other numbers)
		D[findall(isnan.(D))] .= Inf
	
		# Assign each data point to closest median (track number of changes labels)
		changes = 0
		for i in 1:n
			(~,y_new) = findmin(D[i,:])
			changes += (y_new != y[i])
			y[i] = y_new
		end
	
		# Optionally visualize the algorithm steps
		if doPlot && d == 2
			clustering2Dplot(X,y,W)
			sleep(.1)
		end
	
		# Find mean of each cluster
		for c in 1:k
			W[c,:] = median(X[y.==c,:],dims=1)
		end
	
		# Optionally visualize the algorithm steps
		if doPlot && d == 2
			clustering2Dplot(X,y,W)
			sleep(.1)
		end
	
		# Compute the kmeans error
		l1error = l1KMediansError(X, y, W);
	
		@printf("Running k-means, changes = %d, k-means error = %.4f \n", changes, l1error)
	end
	
	function predict(Xhat)
		(t,d) = size(Xhat)
	
		D = l1Distance(Xhat,W)
		D[findall(isnan.(D))] .= Inf
	
		yhat = zeros(Int64,t)
		for i in 1:t
			(~,yhat[i]) = findmin(D[i,:])
		end
		return yhat
	end
	
	return PartitionModel(predict,y,W)
end
	