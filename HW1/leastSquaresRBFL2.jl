include("misc.jl")

function rbfBasis(X1, X2, sigma)
    
    # Create the matrix containing L2 distance squared
    D = distancesSquared(X1, X2);

    # Compute the basis
    Z = exp.(-1 .* D / (2*sigma^2));

    Z;
end

using LinearAlgebra

function leastSquaresRBFL2(X,y; lambda=1, sigma=1)

    (n, d) = size(X);

	# Change the input samples basis to RBF
	Z = rbfBasis(X, X, sigma);
    
	# Find regression weights using RBF basis and L2 norm (Solution found form 340)
    V = inv(Z'*Z + lambda * I(size(Z)[2])) * Z' * y;

	# Make linear prediction function
    function predict(Xtilde)
        Ztilde = rbfBasis(Xtilde, X, sigma);
        ytilde = Ztilde * V;
        ytilde;
    end

	# Return model
	return RBFModel(predict, V, X, sigma)
end
