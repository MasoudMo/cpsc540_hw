# Load data
using JLD
X = load("clusterData2.jld","X")
using Printf

# K-means clustering
k = 4
include("l1KMedians.jl")
include("l1KMediansError.jl")

lowest_error = 9999999;

for i in 1:50
    model = l1KMedians(X,k,doPlot=false)
    y = model.predict(X)
    # Compute error
    error = l1KMediansError(X, y, model.W);
    if error < lowest_error
        global lowest_error = error;
        global best_y = y;
        global best_W = model.W;
    end
end

@printf("Lowest L1 Error is: %.4f", lowest_ssd);
include("clustering2Dplot.jl")
clustering2Dplot(X,best_y,best_W)