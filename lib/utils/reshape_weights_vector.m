function W = reshape_weights_vector(Weights, n_input, n_hidden, n_output)
%RESHAPE_WEIGHTS_VECTOR 
% Reshape a 1-D vector of Weights into cell array with two Weights matrices. 
% For an ANN with 1 hidden layer.

W = cell(2, 1);
W{1} = reshape(Weights(1 : (n_input+1)*n_hidden), n_hidden, n_input+1);
W{2} = reshape(Weights(n_output*n_hidden+1 : end), n_output, n_hidden+1);

end