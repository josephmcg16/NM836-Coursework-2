function h = predict(X, W, act_fun)
%%hyp - calculates the output of a neural network for multiple samples
% inputs:
%           W - cell array of weights matrices for each layer
%           X - matrix of training samples (input layer neurons)
%           act_fun - nonlinear activation function
% ouputs:
%           h - output layer neurons (prediction) for each training sample
addpath('lib')
%% activation functions
if act_fun == "sigmoid"
    phi = @sigmoid;
elseif act_fun == "relu"
    phi = @relu;
end

%% feed-forward network
% activation in first hidden-layer
a{1} = phi([ones(size(X, 1), 1), X] * W{1}');

% activation in hidden-layer
a{2} = phi([ones(size(a{1}, 1), 1), a{1}] * W{2}');
h = a{2};

end