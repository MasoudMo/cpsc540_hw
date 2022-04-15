include("misc.jl") # Includes mode function and GenericModel typedef
include("studentT.jl")


function tda(X,y,k)
  # Implementation of TDA

  # Center the data first
  Xmean = mean(X, dims=1)
  X = X .- Xmean

  # Extract number of samples and their dimension
  d = size(X, 2)
  n = size(X, 1)

  # Initialize array containing the studentT dist for each class
  subModel = Array{DensityModel}(undef,k)
  pis = zeros((k, 1))

  # Go through data and create the dist object for each cass
  for c in 1:k
    pis[c] = sum(y .== c) / n
    subModel[c] = studentT(X[y.==c, :])
  end

  function predict(Xhat)

    # Center the data
    Xhat = Xhat .- Xmean

    t = size(Xhat, 1)
    probs = zeros((t, k))

    # For each sample, find its probability under the class parameters
    for c in 1:k
        probs[:, c] = log(pis[c]) .+ log.(subModel[c].pdf(Xhat))
    end
    yhat = mapslices(argmax, probs, dims=2)

    return yhat

  end

  return GenericModel(predict)
end