include("misc.jl")
include("findMin.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 .+ exp.(-yXw)))
	g = -X'*(y./(1 .+ exp.(yXw)))
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] .= -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict,W)
end

function softmaxObj(W, X, y, d, k)
	# Reshape the weights
	W = reshape(W, d, k);

	n = size(y)[1];

	# Compute intermediate components
	XW = X*W;
	exp_a = exp.(X*W);
	denom_norm = sum(exp_a, dims=2);
	s = broadcast(/, exp_a, denom_norm);

	# Objective
	lse = log.(denom_norm);
	XW_yi = [XW[i, y[i]] for i in 1:n];
	f = sum(-1*XW_yi + lse);

	# Gradient (non-vectorized)
	g = zeros(d, k)
	for c in 1:k
		for j in 1:d
			for i in 1:n
				if y[i] == c
					g[j, c] += X[i, j] * (s[i, c] - 1)
				else
					g[j, c] += X[i, j] * s[i, c]
				end
			end
		end
	end

	g = reshape(g, d*k, 1)

	return (f,g)
end

# Multi-class classifier with Softmax
function softmaxClassifier(X, y)
	(n, d) = size(X);

	# Initialize weights
	k = maximum(y)
	W = zeros(d*k, 1);

	funObj(W) = softmaxObj(W, X, y, d, k);

	W = findMin(funObj, W, verbose=false, derivativeCheck=true)
	W = reshape(W, d, k)

	# Make linear prediction function
	predict(Xhat) = mapslices(argmax,Xhat*W,dims=2)

	return LinearModel(predict, W)
end