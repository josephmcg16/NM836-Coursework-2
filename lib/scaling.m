function [X1, X2, SCALE, CENTER] = scaling(X1, X2, range)
%% scaling features
CENTER(1, :) = min(X1);
SCALE(1, :) = max(X1) - min(X1);

CENTER(2, :) = min(X2);
SCALE(2, :) = max(X2) - min(X2);
X1 = 2 * ((X1 - CENTER(1, :))./SCALE(1, :)) - 1;
X2 = 2 * ((X2 - CENTER(2, :))./SCALE(2, :)) - 1;
return