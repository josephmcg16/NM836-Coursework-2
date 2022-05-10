function [W, w_vec] = weights_init(layers)
%WEIGHTS_INIT

W = cell(1,length(layers)-1);
w_vec = [];
for l = 1:length(layers)-1
    U = rand(layers(l+1), layers(l)+1);  % uniform dist
    if l == length(layers)-1
        % linear activation (output layer)
        W{l} = normalize(U, "range",  [-1, 1]);
    else
        % nonlinear activation (hidden layers)
        W{l} = normalize(U, "range",  [-1/sqrt(layers(l)+1), 1/sqrt(layers(l+1))]);
    end
    w_vec = [w_vec; W{l}(:)];
end

% reshape cell of matrices for each layer into a tall vector