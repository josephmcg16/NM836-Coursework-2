function [grad_cost] = backprop( ...
    y_hat, W, w_vec, z, a, y, act_fun, lmd, n, m, inds_bias)
%BACKPROP back-propogation of errors algorithm
% TODO :
%               - Profile code and find any unecessary computations
%               - Vectorize where possible
%               - Generalize for any number of layers
% Inputs:
%       W       - cell array of weights matrices for each layer
%       act_fun - nonlinear activation function
%       y_hat   - output layer neurons (prediction) for each training ...
%                 ... sample
%       a       - cell array of activation neurons (state values) for ...
%                 ... each layer for each training sample
%       z       - cell array of linear mapping neurons (unit values) ... 
%                 ... for each layer for each training sample
%
%% INIT --------------------------------------------------------------------
if act_fun == "sigmoid"
    %phi = @sigmoid;  % hidden layer activation
    phi_grad = @sigmoid_grad;
end

m1 = length(W{1}(:));
m2 = length(W{2}(:));

grad_loss = zeros(n, m);

%% BACKPROPOGATION AlGORITHM ----------------------------------------------
% loss gradient wrt W{1}
for i = 1:size(W{1}, 2)
    for j = 1:size(W{1}, 1)
        index = size(W{1}, 1) * (i-1) + j;
        grad_loss(:, index) = - (y-y_hat) .* W{2}(j+1) .* ...
                              phi_grad(z{2}(:, j+1)) .* a{1}(:, i);
    end
end  

% loss gradient wrt W{2}
grad_loss(:, m1+1:end) = -(y-y_hat) .* a{2};

% overall (regularised) cost gradient
m = length(setdiff(1:m1+m2, inds_bias));

% boolean array, 0 when bias weight, 1 otherwise
I = true(length(w_vec), 1);
I(inds_bias) = 0;

grad_cost = (1/n)*sum(grad_loss)' + ...
            lmd/m .* w_vec .* I;
return
