using Distributions

function sampleBernoulli(t, theta=0.12)

    # Initialize array holding samples
    samples = zeros(t);

    # Produce t samples
    for i in 1:t
        gaussian_sample = randn();
        sample_cdf = cdf(Normal(), gaussian_sample)

        if sample_cdf <= theta
            samples[i] = 1;
        end
    end

    return samples

end

