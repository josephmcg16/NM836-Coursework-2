function [J, grad] = nnCostFunction(Weights, n_input, n_hidden, n_output, X, y, lambda, act_fun)
%%NNCOSTFUNCTION - Tikhonov variation of the regularised cost function...
%... for an ANN with 1 hidden layer.
% 'predict.m' function handles network prediction and includes the biases
% automatically
addpath('lib/utils', 'lib/activations')
%% reshape Weights vector into cell array
W = reshape_weights_vector(Weights, n_input, n_hidden, n_output);

%% Cost Function
[h, z, a] = predict(X, W, act_fun);  % feed-forward NN prediction

J = (1/size(X, 1)) * sum(sum(-y .* log(h) - (1-y) .* log(1 - h))) ...
    + lambda/(2*length(Weights)) * sum(Weights .^2);

%% Cost Function gradient
% definately a better way to do this vectorised. This works for now though.
if act_fun == "sigmoid"
    phi_grad = @sigmoid_grad;
elseif act_fun == "relu"
    phi_grad = @relu_grad;
end


dJdaL = y./h - (1-y)./(1-h);

dJdW1 = zeros(size(X, 1), size(W{1}, 1), size(W{1}, 2));
for j = 1:1:size(W{1}, 1)
    for k = 1:size(W{1}, 2)
        if k == 1
                dJdW1(:, j, k) = phi_grad(z{1}(:, j)) .* sum(ones(size(X, 1), size(W{2}, 1)) .* W{2}(:, k)' .* phi_grad(z{2}) .* dJdaL, 2);
        else
            dJdW1(:, j, k) = X(:, k-1) .* phi_grad(z{1}(:, j))  .* sum(ones(size(X, 1), size(W{2}, 1)) .* W{2}(:, k)' .* phi_grad(z{2}) .* dJdaL, 2);
        end
    end
end
dJdW1 = sum(dJdW1);

dJdW2 = zeros(size(X, 1), size(W{2}, 1), size(W{2}, 2));
for j = 1:1:size(W{2}, 1)
    for k = 1:size(W{2}, 2)
        if k == 1
            dJdW2(:, j, k) = phi_grad(z{2}(:, j)) .* dJdaL(:, j);
        else
            dJdW2(:, j, k) = a{1}(:, k-1) .* phi_grad(z{2}(:, j)) .* dJdaL(:, j);
        end
    end
end
dJdW2 = sum(dJdW2);

grad = [dJdW1(:); dJdW2(:)] + (lambda / length(Weights)) .* Weights;
end