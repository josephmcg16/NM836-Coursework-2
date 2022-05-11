function MAPE = mape(X,Y)
%MAPE compute Mean Absolute Percentage Error (MAPE) of X and target Y
eps = 1e-7;  % incase X close to zero

X = X(X>eps);
Y = Y(X>eps);

MAPE = 100/size(X, 1) .* sum(abs((Y-X) ./ X));
end

