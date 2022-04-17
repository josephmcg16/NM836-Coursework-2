function [J, grad] = nnCostFunction(Weights, n_input, n_hidden, n_output, X, y, lambda, act_fun)
%%NNCOSTFUNCTION - Tikhonov variation of the regularised cost function... 
%... for an ANN with 1 hidden layer.
% 'predict.m' function handles network prediction and includes the biases
% automatically
addpath('lib/utils')
%% reshape Weights vector into cell array
W = reshape_weights_vector(Weights, n_input, n_hidden, n_output);

%% Cost Function
h = predict(X, W, act_fun);  % feed-forward NN prediction

J = (1/size(X, 1)) * sum(sum(-y .* log(h) - (1-y) .* log(1 - h))) ...
    + lambda/(2*length(Weights)) * sum(Weights .^2);

grad = zeros(length(Weights), 1);
return