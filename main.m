function main()
%%MAIN script for error estimation of different models

%% INIT -----------------------------------------------------------
rng(123);

addpath('Data', 'lib', 'lib/utils')

Trainset = load('Trainset.mat').Trainset;

%% NN parameters ----------------------------------------------------------
n = size(Trainset, 1);                  % number of samples
X = Trainset(:, setdiff(1:end, 35));    % input features
y = Trainset(:, 35);                    % target variable

act_fun = "sigmoid";                    % unit activation

% OPTIMISER SETTINGS
MaxIterLoop = 10;                       % max iterations for all models
MaxIterBest = 100;                      % max iterations for best models

% LAYERS ------------------------------------------------------------------
s1 = size(X, 2);
s2_range = [20, 40]; %, 70, 80, 100];
s3 = size(y, 2);

% HYPERPARAMETERS SEARCH SPACE---------------------------------------------
layers_range = [ones(1, length(s2_range)).*s1; s2_range; ...
                ones(1, length(s2_range)) .* s3]';

lmd_range = logspace(-4, 1, 6);

% K-FOLDS -----------------------------------------------------------------
outer_folds = 2;
inner_folds = 2;

% ERROR METRICS -----------------------------------------------------------
MAE = zeros(1, outer_folds);
MAPE = zeros(1, outer_folds);
REP = zeros(1, outer_folds);
PPMC = zeros(1, outer_folds);

%% MAIN LOOP --------------------------------------------------------------
inds = crossvalind('Kfold', n, outer_folds);
tic
for k = 1:outer_folds
    % train-test split
    test = inds == k;
    train = ~test;
    Xtr = X(train, :);
    Xtest = X(test, :);
    ytr = y(train);
    ytest = y(test);

    % TRAIN MODELS --------------------------------------------------------
    [
     ~, W_opt, cost_opt_his, lmd_best, layers_best...
    ] = trainmodel( ...
                   Xtr, ytr, act_fun, layers_range, lmd_range, ...
                   inner_folds, MaxIterLoop, MaxIterBest ...
                   );

    % TEST BEST MODEL -----------------------------------------------------
    % normalize data in the range (0, 1)
    [Xtest, X_centerValue, X_scaleValue] = normalize(Xtest,"range");
    
    % NETWORK PREDICTION --------------------------------------------------
    y_hat = predict(Xtest, W_opt, act_fun, layers_best);
    
    % DE-NORMALISE --------------------------------------------------------
    y_hat = y_hat .* y_scaleValue + y_centerValue;

    Xtest = Xtest .* X_scaleValue + X_centerValue;

    % CALCULATE ERROR FOR THE SPLIT ---------------------------------------
    MAE(k) = mae(ytest, y_hat);
    MAPE(k) = mape(ytest, y_hat);
    REP(k) = rep(ytest, y_hat);
    PPMC(k) = corr(ytest, y_hat);

    % CHECKPOINT ----------------------------------------------------------
    save('checkpoints/metrics.mat', 'MAE', 'MAPE', 'REP', 'PPMC')
    save(sprintf('checkpoints/model%i.mat', k), ...
        'W_opt', 'cost_opt_his', 'lmd_best', 'layers_best', ...
        'Xtr', 'Xtest', 'ytr', 'ytest')

    % CONSOLE OUTPUT ------------------------------------------------------
    fprintf("-----------------------------------------------------\n" + ...
            "Split %i\tMAE : %3.3f \t\tMAPE : %3.3f \tREP : %3.3f \t" + ...
            "PPMC : %3.3f\n", ...
            k, MAE(k), MAPE(k), REP(k), PPMC(k));
    toc
end