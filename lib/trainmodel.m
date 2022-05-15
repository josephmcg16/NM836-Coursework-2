function [w_vec_opt, W_opt, cost_opt_his, lmd_best, layers_best] = trainmodel( ...
    Xtr, ytr, act_fun, layers_range, lmd_range, kfolds, ...
    MaxIterLoop, MaxIterBest)

%%TRAINMODEL Find best model for a given ANN architecture.
% Splits training sets into k-folds of learning and validation sets to ...
% ... find best model hyperparameters using a global grid search
%
% INPUTS :
%       Xtr             -
%       ytr             -
%       layers_range    -
%       lmd_range       -
%       act_func        -
%       k_folds         -
% OUTPUTS :
%       w_vec_opt       -
%       J_opt           -
%       lmd_best        -
%       layers_best     -

%% CHECKS -----------------------------------------------------------------

if size(Xtr, 1) ~= size(ytr, 1)
    error("Number of samples not consistant for features and labels")
end
%% INIT -------------------------------------------------------------------
% ensure MATLAB root is git repository root
addpath('lib', 'lib/utils', 'lib/activations')

% optimizer options for each loop                                              
options.MaxIter = MaxIterLoop;  % max iterations for each fmincg call

%% HYPERPARAMETERS DOE ----------------------------------------------------
DOE = hyperparams_doe(lmd_range, layers_range);

%% NORMALIZE TRAINING DATA in range (0, 1)
Xtr = normalize(Xtr,"range");
ytr = normalize(ytr,"range");

%% MAIN LOOP -------------------------------------------------------------
inds = crossvalind('Kfold', size(Xtr,1), kfolds);
err_best = +inf;
for i = 1:size(DOE, 1)
    lmd = DOE(i, 1);
    layers = DOE(i, 2:end);
    err = 0;
    for j = 1:kfolds
        % learning and validation split -----------------------------------
        le_inds = inds == j;
        va_inds = ~le_inds;
 
        Xle = Xtr(le_inds, :);
        Xva = Xtr(va_inds, :);

        yle = ytr(le_inds);
        yva = ytr(va_inds);
        
        % init cost function-----------------------------------------------
        [~, w_vec] = weights_init(layers);

        cost_func = @(w) nnCostFunction(w, Xle, yle, lmd, act_fun, layers);

        % optimise weights ------------------------------------------------
        w_vec_opt = fmincg(cost_func,w_vec, options);
        W_opt = reshape_weights_vector(w_vec_opt, layers);
        
        % network prediction ----------------------------------------------
        y_hat = predict(Xva, W_opt, act_fun, layers);
        
        % aggregate error for each leaning / validation split--------------
        err = err + mse(y_hat, yva);
%         fprintf("Inner Split : %i \n", j)
    end

%     fprintf("Sum Err : %3.3f \tlmd : %3.3E \t s2 : %3.3f \n\n", ...
%         err, lmd, layers(2))
    
    % update the best lmd & best error for each split
    if err < err_best
       lmd_best = lmd;
       layers_best = layers;
       err_best = err;
    end        
end

%% TRAIN BEST MODEL -------------------------------------------------------
% init cost function ------------------------------------------------------
[~, w_vec] = weights_init(layers_best);
        
cost_func = @(w) nnCostFunction( ...
    w, Xtr, ytr, lmd_best, act_fun, layers_best);

% optimise weights --------------------------------------------------------
options.MaxIter = MaxIterBest;
[w_vec_opt, cost_opt_his, ~] = fmincg(cost_func,w_vec, options);
W_opt = reshape_weights_vector(w_vec_opt, layers_best);

return
