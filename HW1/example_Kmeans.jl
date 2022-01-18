# Load data
using JLD
X = load("clusterData2.jld","X")
using Printf

# K-means clustering
k = 4
include("kMeans.jl")
include("kMeansError.jl")

lowest_ssd = 9999999;

for i in 1:50
    model = kMeans(X,k,doPlot=false)
    y = model.predict(X)
    # Compute error
    ssd = kMeansError(X, y, model.W);
    if ssd < lowest_ssd
        global lowest_ssd = ssd;
        global best_y = y;
        global best_W = model.W;
    end
end

@printf("Lowest SSD is: %.4f", lowest_ssd);
include("clustering2Dplot.jl")
clustering2Dplot(X,best_y,best_W)