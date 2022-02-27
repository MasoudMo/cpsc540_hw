using JLD

include("sampleBernoulliFromGaussian.jl")

# Number of samples vary between 1 to 100,000
t = 1:100:100100;
theta = 0.12;

expected_values = zeros(0)
for i in t
    samples = sampleBernoulli(convert(Int32, i), theta);
    
    # Apply function to all samples
    samples = map(x -> ifelse(x == 1, -5, 1), samples);

    # Estimate expected value
    append!(expected_values, 1/i * sum(samples));
end

print(last(expected_values))

using PyPlot
figure()
plot(t, expected_values, "r")
xlabel("Number of Samples")
ylabel("Expected Value Estimation")
display(gcf())