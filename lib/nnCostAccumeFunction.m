function [J, grad] = nnCostAccumeFunction(w_vec, n_input, n_hidden, n_output, X, y, lambda, act_fun)
%%NNCOSTFUNCTION - Tikhonov variation of the regularised cost function...
%... for an ANN with 1 hidden layer.
% 'predict.m' function handles network prediction and includes the biases
% automatically
addpath('lib/utils', 'lib/activations')
%% INIT -------------------------------------------------------------------
n = size(X, 1);                             % num training samples
W = reshape_weights_vector( ...
    w_vec, [n_input, n_hidden, n_output]);

if act_fun == "sigmoid"
    phi = @sigmoid;                         % hidden layer activation
end

%% FEED-FORWARD NETWORK ---------------------------------------------------
% first layer
b{1} = ones(size(X, 1), 1);
a{1} = [b{1}, X];

% hidden layer
b{2} = ones(size(X, 1), 1);                 % bias term
z{2} = a{1} * W{1}';                        % linear mapping

[phi_z2, phi_grad_z2] = phi(z{2});          % saves computation
a{2} = [b{2}, phi_z2];                      % activation units

% output-layer
z{3} = a{2} * W{2}';                        % linear mapping
a{3} = z{3};                                % linear (no) activation

y_hat = a{3};                               % network prediction

%% Cost Function - L2 Regularised with Quadratic Loss
L = 1/2 * (y-y_hat).^2;  % loss function

% do not apply regularization to the biases
inds_bias = [1:n_hidden, n_hidden*(n_input+1)+1:n_hidden* ...
    (n_input+1)+n_output];
I = ones(length(w_vec), 1);
I(inds_bias) = 0;
m = length(w_vec) - length(inds_bias);

% calculate network cost
J = 1/n * sum(L) + lambda/m * sum(I .* w_vec .^2);  % network cost

%% Cost Function gradient - backwards propogation
% init
grad_W1 = zeros(size(W{1}));
grad_W2 = zeros(size(W{2}));

% loop round samples
for p = 1:n
    a1 = a{1}(p, :);    % input layer unit
    a2 = a{2}(p, :);    % hidden layer unit
    a3 = a{3}(p, :);    % prediction unit
    
    % output layer
    delta_output = a3 - y(p, :);

    % hidden layer
    delta_hidden = (delta_output * W{2}(:, 2:end)) .* phi_grad_z2(p, :);

    % calculate gradients
    grad_W2 = grad_W2 + delta_output' * a2;
    grad_W1 = grad_W1 + delta_hidden' * a1;
end

grad_W1 = grad_W1 / n;
grad_W2 = grad_W2 / n;

grad = [grad_W1(:); grad_W2(:)] + lambda/m .* I .* w_vec .^2;