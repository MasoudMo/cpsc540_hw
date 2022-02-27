
function sampleBernoulli(t, theta=0.12)

    # Initialize array holding samples
    samples = zeros(t);

    # Produce t samples
    for i in 1:t
        uniform_sample = rand();

        if uniform_sample <= theta
            samples[i] = 1;
        end
    end

    return samples

end
