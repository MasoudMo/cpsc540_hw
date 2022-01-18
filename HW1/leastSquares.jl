include("misc.jl")

function leastSquares(X,y)

	# Add a columns of 1's to X for the y intercept
	(n, d) = size(X)
	X = hcat(ones(n, 1), X)

	# Find regression weights minimizing squared error
	w = (X'*X)\(X'*y)

	# Make linear prediction function
	predict(Xtilde) = hcat(ones(size(Xtilde)[1], 1), Xtilde)*w

	# Return model
	return LinearModel(predict,w)
end
