function DOE = hyperparams_doe(lmd_range, layers_range)
%%HYPERPARAMSDOE Construct array of samples of a vector for the network ...
% ... hyperparameters.
%
% Inputs :
%       lmd_range       - Vector of possible lmd regularisation parameters
%       layers_range    - Samples of possible vectors for each number ...
%                         ... layers in the ANN
% Outputs :
%       DOE             - Array of Samples of each possible combination ...
%                         ... of hyperparameters in the input ranges
%
% MAIN LOOP ---------------------------------------------------------------
DOE = zeros(length(lmd_range) * size(layers_range, 1), ...
    1 + size(layers_range, 2));
for i = 1:length(lmd_range)
    for j = 1:size(layers_range, 1)
        smpl = size(layers_range, 1) * (i-1) + j;
        DOE(smpl, :) = [lmd_range(i), layers_range(j, :)];
    end
end
return