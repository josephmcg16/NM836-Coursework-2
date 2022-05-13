function gradJ_diff = backprop_approx(cost_func, Weights, dWeights)

gradJ_diff = zeros(length(Weights), 1);
for i = 1:length(Weights)
    dJ = cost_func(Weights + dWeights) - cost_func(Weights);
    gradJ_diff(i) = dJ / delta;
end
