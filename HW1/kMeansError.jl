
function kMeansError(X, y, W)
    # Get the dimensions and ensure they are correct
    (nx, dx) = size(X);
    (ny, ) = size(y);
    (nt, dt) = size(W);
    @assert(nx==ny)
    @assert(dx==dt)

    # Initialize error
    ssd = 0;

    # Non-vectorized solution
    for i in 1:nx
        for j in 1:dx
            ssd += (X[i, j] - W[y[i], j])^2;
        end
    end

    ssd;

end
