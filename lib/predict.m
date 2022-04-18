function [h, z, a] = predict(X, W, act_fun)
%%hyp - calculates the output of a neural network for multiple samples
% inputs:
%           W - cell array of weights matrices for each layer
%           X - matrix of training samples (input layer neurons)
%           act_fun - nonlinear activation function
% ouputs:
%           h - output layer neurons (prediction) for each training sample
addpath('lib/activations')
%% activation functions
if act_fun == "sigmoid"
    phi = @sigmoid;
elseif act_fun == "relu"
    phi = @relu;
end

%% feed-forward network
% first hidden-layer
z{1} = [ones(size(X, 1), 1), X] * W{1}';
a{1} = phi(z{1});

% output-layer
z{2} = [ones(size(a{1}, 1), 1), a{1}] * W{2}';
a{2} = phi(z{2});
h = a{2};

end