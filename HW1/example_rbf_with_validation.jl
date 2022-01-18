# Load X and y variable
using JLD
using Printf
using Random
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])

# Compute number of training examples and number of features
(n,d) = size(X)

# Shuffle data
shuffle_perm = randperm(n);
X = X[shuffle_perm];
y = y[shuffle_perm];

# Split the data into training and validation
X_train = X[1:floor(Int, n/2), :];
y_train = y[1:floor(Int, n/2), :];
X_val = X[floor(Int, n/2)+1:end, :];
y_val = y[floor(Int, n/2)+1:end, :];

# Fit least squares model
include("leastSquaresRBFL2.jl")

sigma_vals = 1e-5:0.1:5;
lambda_vals = 1e-5:0.1:5;
lowest_mse = 9999999;
for sigma_val in sigma_vals
    for lambda_val in lambda_vals
        model = leastSquaresRBFL2(X_train, y_train, lambda=lambda_val, sigma=sigma_val);
        yhat = model.predict(X_val);

        # Compute MSE (mean squared error)
        mse = sum((yhat - y_val).^2) / (n/2);
        if mse < lowest_mse
            global lowest_mse = mse;
            global chosen_model = model;
            global chosen_sigma = sigma_val;
            global chosen_lambda = lambda_val
        end
    end
end

@printf("MSE: %.4f with lambda: %.6f and sigma: %.6f", lowest_mse, chosen_lambda, chosen_sigma);


# Plot the validation performance
using PyPlot
figure()
plot(X_val, y_val, "b.")
Xhat = minimum(X_val):maximum(X_val)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = chosen_model.predict(Xhat)
plot(Xhat[:],yhat,"r")
ylim((0,2.5))
display(gcf())

# Plot the test performance
figure()
plot(X,y,"b.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = chosen_model.predict(Xhat)
plot(Xhat[:],yhat,"r")
ylim((0,2.5))
display(gcf())