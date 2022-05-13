function REP = rep(X,Y)
%REP compute relative error percentage of X predicting Y
REP = 100 * sqrt(sum((Y-X).^2) ./ sum(X.^2));
end

