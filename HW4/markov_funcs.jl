include("misc.jl")


function sampleAncestral(p1, pt, s, d)
    # Performs s ancestral samplings of a markov chain for d transitions
    samples = zeros((s, d))

    for simulation in 1:s
        # Initial state sampling
        samples[simulation, 1] = sampleDiscrete(p1)

        # Generate samples ancestrally using the transition probabilities
        for step in 2:d
            samples[simulation, step] = sampleDiscrete(pt[floor(Int, samples[simulation, step-1]), :])
        end
    end

    return samples
end


function marginalCK(p1, pt, d)
    # Finds exact marginal at step d
    marginals = pt' * p1

    for step in 1:d-2
        marginals = pt' * marginals
    end

    return marginals
end


function fb_algorithm(p1, pt, k, d, cond, j)

    # initialize the M matrix
    M = zeros((k, d))
    M[:, 1] = p1

    # initialize phi
    phi = ones((k, d))
    phi[:, 1] = p1
    
    # Conditioning
    phi[:, d] = zeros(k)
    phi[cond, d] = 1

    # Initialize the V matrix
    V = zeros((k, d))
    V[:, d] = phi[:, d]

    # Populate the M matrix
    for step in 2:d
        for xj in 1:k
            for xjprev in 1:k
                M[xj, step] = M[xj, step] + phi[xj, j] * pt[xjprev, xj] * M[xjprev, step-1]
            end
        end
    end

    # Normalizing constant
    Z = 

    # Populate the V matrix
    for step in d-1:-1:1
        for xj in 1:k
            for xjnext in 1:k
                V[xj, step] = V[xj, step] + phi[xj, j] * pt[xj, xjnext] * V[xjnext, step+1]
            end
        end
    end

    prob = M[:, j] .* V[:, j] ./ sum(M[:, j] .* V[:, j])
    return prob
end