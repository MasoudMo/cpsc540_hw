# Load X and y variable
using JLD, Printf, Statistics
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
y[y.==2] .= 0
ytest[ytest.==2] .= 0
(n,d) = size(X)
X = [ones(n,1) X]
(n,d) = size(X)
Xtest = [ones(size(Xtest)[1],1) Xtest]

X = X ./ 255
Xtest = Xtest ./ 255


# Choose network structure and randomly initialize weights
include("newNeuralNet_mnist35.jl")
nHidden = [350, 128, 64]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
maxIter = 50000
stepSize = 1e-3
samples_per_batch = 361
patience = 3
prev_err = 9999
for t in 1:maxIter

    # The stochastic gradient update:
    i = rand(1:n, samples_per_batch)
    (f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden,initialize=(i == 1))
    global w = w - stepSize*g

    # Every few iterations, plot the data/model:
    if (mod(t-1,round(100)) == 0)
        yhat = NeuralNet_predict(w,Xtest,nHidden) .> 0.5
        err = sum(yhat .!= ytest)/size(Xtest,1)
        
        if err < prev_err
            patience = 3
            prev_err = err
        else
            if patience == 1
                stepSize /= 2
                @printf("Reduced step size to %.4f", stepSize)
                patience = 3
            else
                patience -= 1
            end
        end

        @printf("Training iteration = %d, error rate = %.2f\n",t-1,err)
    end
end
