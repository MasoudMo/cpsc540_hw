mutable struct LinearModel
	predict # Funcntion that makes predictions
	w # Weight vector
end

mutable struct RBFModel
	predict # Funcntion that makes predictions
	V # Weight vector
	X # Training data
	sigma
end

mutable struct CompressModel
	compress # Function that compresses
	expand # Function that de-compresses
	W # weight matrix
end

# Return squared Euclidean distance all pairs of rows in X1 and X2
function distancesSquared(X1,X2)
	(n,d) = size(X1)
	(t,d2) = size(X2)
	@assert(d==d2)
	return X1.^2*ones(d,t) .+ ones(n,d)*(X2').^2 .- 2X1*X2'
end

# Return L1 distance of all pairs of rows in X1 and X2
using LinearAlgebra
function l1Distance(X1,X2)
	(n,d) = size(X1)
	(t,d2) = size(X2)
	@assert(d==d2)

	# Create output array
	D = zeros(n, t);
	for i in 1:n
		for j in 1:t
			D[i, j] = norm(X1[i, :] .- X2[j, :], 1);
		end
	end
	D;
end

### A function to compute the gradient numerically
function numGrad(func,x)
	n = length(x);
	delta = 2*sqrt(1e-12)*(1+norm(x));
	g = zeros(n);
	e_i = zeros(n)
	for i = 1:n
		e_i[i] = 1;
		(fxp,) = func(x + delta*e_i)
		(fxm,) = func(x - delta*e_i)
		g[i] = (fxp - fxm)/2delta;
		e_i[i] = 0
	end
	return g
end

### Check if number is a real-finite number
function isfinitereal(x)
	return (imag(x) == 0) & (!isnan(x)) & (!isinf(x))
end
