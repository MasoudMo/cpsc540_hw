include("misc.jl") # Includes mode function and GenericModel typedef
using Distributions


function gda(X,y,k)
  # Implementation of GDA

  # Extract number of samples and their dimension
  d = size(X, 2)
  n = size(X, 1)

  # Initialize parameters
  sigmas = zeros((k, d, d))
  mus = zeros((k, d))
  pis = zeros((k, 1))

  # Go through data and extract required parameters 
  # for each class
  for c in 1:k
    nc = sum(y .== c)
    pis[c] = nc / n
    mus[c, :] = sum(X[y.==c, :], dims=1) ./ nc
    sigmas[c, :, :] = transpose(X[y.==c, :] .- mus[c]) * (X[y.==c, :] .- mus[c]) ./ nc
  end

  function predict(Xhat)
    t = size(Xhat, 1)
    yhat = zeros(t)

    # For each sample, find its probability under the class parameters
    for i in 1:t
        max_prob = -9999999
        max_prob_class = 0
        for c in 1:k
            d = MvNormal(mus[c, :], sigmas[c, :, :])
            prob = log(pis[c]) + logpdf(d, Xhat[i, :])
            if prob > max_prob
                max_prob = prob
                max_prob_class = c
            end
        end
        yhat[i] = max_prob_class
    end

    return yhat

  end

  return GenericModel(predict)
end