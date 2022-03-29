
function viterbiDecode(p1, pt, k, d)
    # Initialize the M matrix
    M = zeros((k, d))
    M[:, 1] = p1

    # Initialize the backtracking matrix
    B = zeros((k, d))

    # Fill out the matrices for each step
    for step in 2:d

        # Go through all possible values of current state
        for xj in 1:k
            max_prob = -9999999
            max_c = 0

            # Find the previous state giving highest M for current step
            for xjprev in 1:k
                prob = pt[xjprev, xj] * M[xjprev, step-1]
                if max_prob < prob
                    max_prob = prob
                    max_c = xjprev
                end
            end

            # populate the DP matrices
            M[xj, step] = max_prob
            B[xj, step] = max_c
        end
    end

    # Initialize decoded chain
    decoded_chain = zeros(d)

    # Perform backtracking
    max_xj = argmax(M[:, d])
    decoded_chain[d] = max_xj


    for step in d:-1:2
        decoded_chain[step-1] = B[max_xj, step]
        max_xj = argmax(M[:, step-1])
    end

    return decoded_chain
end