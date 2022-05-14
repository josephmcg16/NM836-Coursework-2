function [cost, grad_cost] = nnCostFunction( ...
    w_vec, X, y, lmd, act_fun, layers)
%%NNCOSTFUNCTION - Tikhonov variation of the regularised cost function...
%... for an ANN with 1 hidden layer.

%% INIT -------------------------------------------------------------------
addpath('lib', 'lib/activations', 'lib/utils')

n = size(X, 1);                                 % num training samples
W = reshape_weights_vector(w_vec, layers);      % get weights matrices
n_layers = length(layers);

% do not do regularisation for biases
inds_bias = [1:layers(2), layers(2)*(layers(1)+1)+1:layers(2)* ...
    (layers(1)+1)+layers(3)];
I = true(length(w_vec), 1);
I(inds_bias) = 0;

m = length(w_vec) - length(inds_bias);

% nonlinear activation function
if act_fun == "sigmoid"
    phi = @sigmoid;
elseif act_fun == "hyptan"
    phi = @hyptan;
elseif act_fun == "relu"
    phi = @relu;
end

%% FORWARD PROPOGATION ----------------------------------------------------
% init
a = cell(1, n_layers);
z = cell(1, n_layers);

% first layer
b = ones(size(X, 1), 1);                % bias term
a{1} = [b, X];                          % add bias

% hidden layer
z{2} = a{1} * W{1}';                    % linear mapping
[phi_z2, phi_grad_z2] = phi(z{2});      % nonlinear activation
a{2} = [b, phi_z2];                     % activation units
% output-layer
z{3} = a{2} * W{2}';                    % linear mapping (identity)
a{3} = z{3};                            % identify function

% network output
y_hat = a{3};

%% NEURAL NETWORK COST FUNCTION - L2 REGULARISED MSE ----------------------
cost = 1/(2*n) * sum((y-y_hat).^2) + ...
       lmd/m * sum(w_vec .* I .^2);

%% BACKWARDS PROPOGATION --------------------------------------------------
<<<<<<< Updated upstream
=======
%% BACKWARDS PROPOGATION --------------------------------------------------
% init
>>>>>>> Stashed changes
grad_cost = zeros(length(w_vec), 1);

% output prediction error
deltaL = y-y_hat;

% loss gradient wrt W{1}
<<<<<<< Updated upstream
grad_W1 = (deltaL * W{2}(:, 2:end) .* phi_grad_z2)' * a{1};
=======
grad_W1 = (-deltaL * W{2}(:, 2:end) .* phi_grad_z2)' * a{1};
>>>>>>> Stashed changes
grad_cost(1:length(W{1}(:))) = grad_W1(:);

% loss gradient wrt W{2}
grad_W2 = -deltaL' * a{2};
grad_cost(length(W{1}(:))+1:end) = grad_W2;

% regularised cost gradient
grad_cost = (1/n).*grad_cost + ...
            lmd/m .* w_vec .* I;

return
