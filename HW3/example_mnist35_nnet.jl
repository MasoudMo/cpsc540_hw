# Load X and y variable
using JLD, Printf, Statistics
using Flux
using Flux.Optimise: update!
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
y[y.==2] .= 0
ytest[ytest.==2] .= 0
(n,d) = size(X)

data_std = std(X)
data_mean = mean(X)

X = X.-data_mean ./ data_std
Xtest = Xtest.-data_mean ./ data_std

# Model definition using flux
model = Chain(Dense(d, 30, tanh, init=Flux.kaiming_uniform), 
              Dense(30, 20, tanh, init=Flux.kaiming_uniform), 
              Dense(20, 5, tanh, init=Flux.kaiming_uniform),
              Dense(5, 1, sigmoid, init=Flux.kaiming_uniform))
loss(x, y) = Flux.binarycrossentropy(model(x), y)
ps = params(model)

# Define the optimizer
stepSize = 1e-2
opt = Descent(stepSize)

# Train with stochastic gradient
maxIter = 50000
samples_per_batch = 361
patience = 3
prev_err = 9999
for t in 1:maxIter

	i = rand(1:n, samples_per_batch)

	grads = Flux.gradient(ps) do 
		loss(transpose(X[i, :]), reshape(y[i], 1, size(y[i], 1)))
	end
	update!(opt, ps, grads)

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(1000)) == 0)
		yhat = model(transpose(Xtest)) .> 0.5
		err = sum(yhat .!= reshape(ytest, 1, size(ytest, 1)))/size(Xtest,1)
		@printf("Training iteration = %d, error rate = %.2f\n",t-1,err)

		if err < prev_err
            patience = 3
            prev_err = err
        else
            if patience == 1
                stepSize /= 10
				opt = Descent(stepSize)
                @printf("Reduced step size to %.4f", stepSize)
                patience = 3
            else
                patience -= 1
            end
        end
	end
end
