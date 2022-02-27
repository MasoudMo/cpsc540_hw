# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

## Fit logistic regression model
include("logreg.jl")
model = logReg12(X,y)

## Compute error on test data
yhat = model.predict(Xtest)
err = sum(yhat .!= ytest)/size(Xtest,1)
@printf("Error rate = %.2f\n",err)

@printf("Predicting...\n")

## Compute error on train data
#yhat = model.predict(X)
#err = sum(yhat .!= y)/size(X,1)
#@printf("Error rate = %.2f\n",err)

## Compute error on test data
yhat = model.predict(Xtest)
err = sum(yhat .!= ytest)/size(Xtest,1)
@printf("Error rate = %.2f\n",err)
