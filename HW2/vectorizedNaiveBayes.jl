include("misc.jl") # Includes GenericModel typedef
include("kMeans.jl")
using PyPlot

function vectorizedNaiveBayes(X,y,k)
	# Implementation of VQNB
	(n,d) = size(X)
    
    # Perform kmeans first
    model_3 = kMeans(X[y.==1,:], k)
    z_3 = model_3.predict(X[y.==1, :])
    model_5 = kMeans(X[y.==2, :], k)
    z_5 = model_5.predict(X[y.==2, :])

	# Compute p(y = 1)
	p_y = sum(y.==1)/n

    # Let's create p_zy
    p_zy = zeros(k, 2)
	for c in 1:k
        p_zy[c, 1] = sum(z_3 .== c) / sum(y.==1)
        p_zy[c, 2] = sum(z_5 .== c) / sum(y.==2)
	end   

	# Let's create p_xyz
	p_xyz = zeros(d, 2, k)
	for j in 1:d
        for c in 1:k
            p_xyz[j, 1, c] = sum((X[y.==1,j].==1) .& (z_3.==c))/sum(z_3.==c)
            p_xyz[j, 2, c] = sum((X[y.==2,j].==1) .& (z_5.==c))/sum(z_5.==c)
        end
	end

    # Show learned images
    for c in 1:k
        for i in 1:2
            imshow(reshape(p_xyz[:, i, c],28,28)',"gray")
            display(gcf())
        end
    end
    

	function predict(Xhat)
		(t,d) = size(Xhat)
		yhat = zeros(t)

		for i in 1:t
			# p_yx = p_y*prod(p_x) for the appropriate x and y values
			p_yx = [p_y;1-p_y]
            sum_over_cluster = zeros(2)
            for c in 1:k
                prod_over_features = ones(k, 2)
                for j in 1:d
                    if Xhat[i, j] == 1
                        prod_over_features[c, 1] *= p_xyz[j, 1, c]
                        prod_over_features[c, 2] *= p_xyz[j, 2, c]
                    else
                        prod_over_features[c, 1] *= 1-p_xyz[j, 1, c]
                        prod_over_features[c, 2] *= 1-p_xyz[j, 2, c]
                    end
                end
                sum_over_cluster[1] = sum_over_cluster[1] + p_zy[c, 1] * prod_over_features[c, 1]
                sum_over_cluster[2] = sum_over_cluster[2] + p_zy[c, 2] * prod_over_features[c, 2]
            end
            p_yx[1] *= sum_over_cluster[1]
            p_yx[2] *= sum_over_cluster[2]
            if p_yx[1] > p_yx[2]
				yhat[i] = 1
			else
				yhat[i] = 2
			end
		end
		return yhat
	end

	return GenericModel(predict)
end
