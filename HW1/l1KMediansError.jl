using LinearAlgebra
function l1KMediansError(X, y, W)
    # Get the dimensions and ensure they are correct
    (nx, dx) = size(X);
    (ny, ) = size(y);
    (nt, dt) = size(W);
    @assert(nx==ny)
    @assert(dx==dt)

    # Initialize error
    l1error = 0;

    # Non-vectorized solution
    for i in 1:nx
        for j in 1:dx
            l1error += norm(X[i, j] - W[y[i], j], 1);
        end
    end

    l1error;

end