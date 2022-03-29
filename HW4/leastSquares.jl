using LinearAlgebra
using Printf
include("misc.jl")


function leastSquaresBasis(x,y,p)
	Z = polyBasis(x,p)

	w = (Z'*Z)\(Z'*y)

	predict(xhat) = polyBasis(xhat,p)*w

	return GenericModel(predict)
end

function polyBasis(x,p)
	n = length(x)
	Z = zeros(n,p+1)
	for i in 0:p
		Z[:,i+1] = x.^i
	end
	return Z
end

function leastSquaresEmpiricalBaysis(x, y, max_p, max_sigma, max_lambda)

	marg_likes = zeros((max_p, floor(Int, log2(max_lambda) + 1), floor(Int, log2(max_sigma) + 1)))

	count = 1
	for p in 1:max_p
		for sig in 0:floor(Int, log2(max_sigma))
			for lam in 0:floor(Int, log2(max_lambda))

				# Using powers of 2 for lambda and sigma
				sigma = 2^sig
				lambda = 2^lam

				# Create the polynomial basis
				Z = polyBasis(x, p)

				n, k = size(Z)

				# Find the log marginal likelihood
				Theta = 1/sigma^2 * Z' * Z + lambda * I
				wp = 1/sigma^2 * inv(Theta) * Z' * y
				marg_likes[p, lam+1, sig+1] = k/2*log(lambda) - n * log(sigma) - n * log(sqrt(2*π)) - 1/2 * 
				                              logdet(Theta) + (-1/(2*sigma^2) * norm(Z * wp - y, 2)^2 - lambda/2 * norm(wp, 2)^2)
				count += 1
			end
		end
	end

	max = maximum(marg_likes)
	maxidx = findall(x->x==max, marg_likes)[1]
	sigma = 2^maxidx[3]
	lambda = 2^maxidx[2]
	p = maxidx[1]

	@printf("Chosen P = %f, Chosen Sigma = %f, Chosen Lambda = %f \n", p, sigma, lambda)

end

function polyBasis(x,p)
	n = length(x)
	Z = zeros(n,p+1)
	for i in 0:p
		Z[:,i+1] = x.^i
	end
	return Z
end

