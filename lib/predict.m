function [y_hat, a, z] = predict(X, W, act_fun, layers)
%%PREDICT feed-forward neural network outputs
% Inputs:
%       W               - cell array of weights matrices for each layer
%       X               - matrix of training samples (input layer neurons)
%       act_fun         - nonlinear activation function
%       n_hidden_layers
% Ouputs:
%       y_hat   - output layer neurons (prediction) for each training ...
%                 ... sample
%       a       - cell array of activation neurons (state values) for ...
%                 ... each layer for each training sample
%       z       - cell array of linear mapping neurons (unit values) ... 
%                 ... for each layer for each training sample
%
%% INIT -------------------------------------------------------------------
addpath('lib/activations')

% hidden layer activations
if act_fun == "sigmoid"
    phi = @sigmoid;
    % add extra conditions for more activation functions?
end
n_layers = length(layers);

a = cell(1, n_layers);
z = cell(1, n_layers);

%% FEED-FORWARD PROPOGATION -----------------------------------------------
% first layer
b = ones(size(X, 1), 1);  % bias term
a{1} = [b, X];

for l = 2:n_layers - 1
% hidden layer
z{l} = [b, a{l-1} * W{l-1}'];  % linear mapping
a{l} = [b, phi(z{l}(:, 2:end))];  % activation units
end

% output-layer
z{end} = a{n_layers-1} * W{n_layers-1}';  % linear mapping (identity)
a{end} = z{end};  % identify function

% network prediction
y_hat = a{end};

end