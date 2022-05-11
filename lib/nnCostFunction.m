function [cost, grad_cost] = nnCostFunction( ...
    w_vec, X, y, lmd, act_fun, layers)
%%NNCOSTFUNCTION - Tikhonov variation of the regularised cost function...
%... for an ANN with 1 hidden layer.
%
% 'predict.m' function handles network prediction and includes the biases
% automatically

%% INIT -------------------------------------------------------------------
n = size(X, 1);                                 % num training samples
m = length(w_vec);
W = reshape_weights_vector(w_vec, layers);        % get weights matrices

%% FORWARD PROPOGATION ----------------------------------------------------
[y_hat, a, z] = predict(X, W, act_fun, layers);

%% NEURAL NETWORK COST FUNCTION - L2 REGULARISED MSE ----------------------
inds_bias = [1:layers(2), layers(2)*(layers(1)+1)+1:layers(2)* ...
    (layers(1)+1)+layers(3)];
cost = 1/(2*n) * sum((y-y_hat).^2) + ...
       lmd/m * sum(w_vec(setdiff(1:end, inds_bias)) .^2);

%% BACKWARDS PROPOGATION --------------------------------------------------
[grad_cost] = backprop( ...
    y_hat, W, w_vec, z, a, y, act_fun, lmd, n, m, inds_bias);

return
