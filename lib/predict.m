function y_hat = predict(X, W, act_fun, layers)
%%PREDICT feed-forward neural network outputs
% Inputs:
%       W               - cell array of weights matrices for each layer
%       X               - matrix of training samples (input layer neurons)
%       act_fun         - nonlinear activation function
%       layers          - array containing number of units in each ...
%                         layer (not including biases)
% Ouputs:
%       y_hat   - output layer neurons (prediction) for each training ...
%                 ... sample
%
%% INIT -------------------------------------------------------------------
addpath('lib', 'lib/activations', 'lib/utils')
n_layers = length(layers);

% activation function
if act_fun == "sigmoid"
    phi = @sigmoid;
end

%% FORWARD PROPOGATION ----------------------------------------------------
% init
a = cell(1, n_layers);
z = cell(1, n_layers);

% first layer
b = ones(size(X, 1), 1);  % bias term
a{1} = [b, X];

% hidden layer
z{2} = a{1} * W{1}';  % linear mapping
a{2} = [b, phi(z{2})];  % activation units
% output-layer
z{3} = a{2} * W{2}';  % linear mapping (identity)
a{3} = z{3};  % identify function

% network output
y_hat = a{3};

end