function [J, grad] = nnCostFunction(Weights, n_input, n_hidden, n_output, X, y, lambda, act_fun)
%%NNCOSTFUNCTION - Tikhonov variation of the regularised cost function...
%... for an ANN with 1 hidden layer.
% 'predict.m' function handles network prediction and includes the biases
% automatically
addpath('lib/utils', 'lib/activations')
%% init
n = size(X, 1);  % num training samples
m = length(Weights);

W = reshape_weights_vector(Weights, n_input, n_hidden, n_output);
[y_hat, a, z] = predict(X, W, act_fun);  % feed-forward propogation

%% Cost Function - L2 Regularised with Quadratic Loss
L = 1/2 * (y-y_hat).^2;  % loss function
J = 1/n * sum(L) + lambda/m * sum(Weights .^2);  % network cost

%% Cost Function gradient - backwards propogation
% init
if act_fun == "sigmoid"
    phi = @sigmoid;  % hidden layer activation
    phi_grad = @sigmoid_grad;
end

m1 = length(W{1}(:));
m2 = length(W{2}(:));

grad_loss = zeros(n, m);

% loss gradient wrt W{1}
% TODO: vectorise these loops
% http://cs231n.stanford.edu/slides/2018/cs231n_2018_ds02.pdf
for i = 1:n_input+1
    for j = 1:n_hidden
        index = n_hidden * (i-1) + j;
        grad_loss(:, index) = - (y-y_hat) .* W{2}(j) .* phi_grad(z{2}(:, j)) .* a{1}(:, i);
    end
end

% loss gradient wrt W{2}
grad_loss(:, m1+1:end) = -(y-y_hat) .* a{2};

% overall (regularised) cost gradient
grad = (1/n)*sum(grad_loss)' + lambda/m .* Weights;



