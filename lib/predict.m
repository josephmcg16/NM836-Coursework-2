function [y_hat, a, z] = predict(X, W, act_fun)
%%hyp - calculates the output of a neural network for multiple samples
% inputs:
%           W - cell array of weights matrices for each layer
%           X - matrix of training samples (input layer neurons)
%           act_fun - nonlinear activation function
% ouputs:
%           y_hat - output layer neurons (prediction) for each training sample
%           a - cell array of activation neurons (unit values) for each
%           layer for each training sample
%           z - cell array of linear mapping neurons (unit values) for each layer for
%           each training sample
addpath('lib/activations')
%% activation functions
if act_fun == "sigmoid"
    phi = @sigmoid;
end

%% feed-forward network
% first layer
b{1} = ones(size(X, 1), 1);
a{1} = [b{1}, X];

% hidden layer
b{2} = ones(size(X, 1), 1);  % bias term
z{2} = [b{2}, a{1} * W{1}'];  % linear mapping
a{2} = phi(z{2});  % activation units

% output-layer
z{3} = a{2} * W{2}';  % linear mapping
a{3} = z{3};  % identify function

% network prediction
y_hat = a{3};

end