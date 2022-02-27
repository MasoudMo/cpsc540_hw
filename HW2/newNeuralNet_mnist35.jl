# We use nHidden as a vector, containing the number of hidden units in each layer
# Definitely not the most efficient implementation!

# Function that returns total number of parameters
function NeuralNet_nParams(d,nHidden)

	# Connections from inputs to first hidden layer
	nParams = d*nHidden[1]

	# Connections between hidden layers
	for h in 2:length(nHidden)
		nParams += nHidden[h-1]*nHidden[h]
	end

	# Connections from last hidden layer to output
	nParams += nHidden[end]

end

# Compute squared error and gradient
# for a single training example (x,y)
# (x is assumed to be a column-vector)
function NeuralNet_backprop(bigW,x,y,nHidden; initialize=false)
	(n, d) = size(x)

	nLayers = length(nHidden)

	if initialize
		print("Initializing weights. \n")
	end

	#### Reshape 'bigW' into vectors/matrices
	# - This is not a really elegant way to do things
	# if you want to be really efficient, but for the course
	# it is nice abraction
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)

	if initialize
		W1 .*= sqrt(2.0/size(W1)[2])
	end
	
	ind = nHidden[1]*d
	Wm = Array{Any}(undef,nLayers-1)
	for layer in 2:nLayers
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])

		if initialize
			Wm[layer-1] .*= sqrt(2.0/size(Wm[layer-1])[2])
		end
		
		ind += nHidden[layer]*nHidden[layer-1]
	end
	v = bigW[ind+1:end]

	if initialize
		v .*= sqrt(2.0/size(v)[1])
	end

	#### Define activation function and its derivative
	h(z) = tanh.(z)
	dh(z) = (sech.(z)).^2
    # h(z) = max.(0, z)
    # dh(z) = ifelse.(z.>0, 1, 0)

	#### Forward propagation
	z = Array{Any}(undef,nLayers)
	z[1] = W1*x'
	for layer in 2:nLayers
		z[layer] = Wm[layer-1]*h(z[layer-1])
	end
	yhat = v'*h(z[end])
    yhat = 1 ./ (1 .+ exp.(-1 .* yhat))
	yhat = yhat'

	f = -1/n * sum(y .* log.(yhat) + (1 .- y) .* log.(1 .- yhat))

	#### Backpropagation (the below could be replaced by AD)
	# dr = 1/n * (yhat .- y) ./ (yhat .* (1 .- yhat))
	err = yhat .- y
	
	# Output weights
	Gout = h(z[end]) * err

	Gm = Array{Any}(undef,nLayers-1)
	if nLayers > 1
		# Last Layer of Hidden Weights
		backprop = err' .* (dh(z[end]).*v)
		Gm[end] = backprop*h(z[end-1])'

		# Other Hidden Layers
		for layer in nLayers-2:-1:1
			backprop = (Wm[layer+1]'*backprop).*dh(z[layer+1])
			Gm[layer] = backprop*h(z[layer])'
		end

		# Input Weights
		backprop = (Wm[1]'*backprop).*dh(z[1])
		G1 = backprop*x
	else
		# Input weights
		G1 = err' .* (dh(z[1]).*v)*x
	end

	#### Put gradients into vector
	g = zeros(size(bigW))
	g[1:nHidden[1]*d] = G1
	ind = nHidden[1]*d
	for layer in 2:nLayers
		g[ind+1:ind+nHidden[layer]*nHidden[layer-1]] = Gm[layer-1]
		ind += nHidden[layer]*nHidden[layer-1]
	end
	g[ind+1:end] = Gout

	return (f,g)
end

# Computes predictions for a set of examples X
function NeuralNet_predict(bigW,Xhat,nHidden)
	(t,d) = size(Xhat)
	nLayers = length(nHidden)

	#### Reshape 'bigW' into vectors/matrices
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = Array{Any}(undef,nLayers-1)
	for layer in 2:nLayers
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		ind += nHidden[layer]*nHidden[layer-1]
	end
	v = bigW[ind+1:end]

	# #### Define activation function and its derivative
	h(z) = tanh.(z)
	dh(z) = (sech.(z)).^2
    # h(z) = max.(0, z)
    # dh(z) = ifelse.(z.>0, 1, 0)

	#### Forward propagation on each example to make predictions
	yhat = zeros(t,1)
	for i in 1:t
		# Forward propagation
		z = Array{Any}(undef,1nLayers)
		z[1] = W1*Xhat[i,:]
		for layer in 2:nLayers
			z[layer] = Wm[layer-1]*h(z[layer-1])
		end
		yhat[i] = v'*h(z[end])
	end
	return 1 ./ (1 .+ exp.(-1 .* yhat))
end

