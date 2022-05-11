function W = reshape_weights_vector(Weights, layers)
%RESHAPE_WEIGHTS_VECTOR 
% Reshape a 1-D vector of Weights into cell array with Weights matrices ...
% for each layer in layers.

%% INIT -------------------------------------------------------------------
W = cell(1, length(layers)-1);

%% MAIN LOOP --------------------------------------------------------------
if length(layers) == 3
    n_input = layers(1);
    n_hidden = layers(2);
    n_output = layers(3);

    W{1} = reshape(Weights(1 : (n_input+1)*n_hidden), n_hidden, n_input+1);
    W{2} = reshape(Weights((n_input+1)*n_hidden+1 : end), n_output, n_hidden+1);
else
    error("Reshape not setup yet for other structures of ANN...")
end

return
