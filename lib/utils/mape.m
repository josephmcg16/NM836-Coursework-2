function MAPE = mape(X,Y)
%MAPE compute Mean Absolute Percentage Error (MAPE) of X and target Y
MAPE = 100/size(X, 1) * sum(abs((Y-X) ./ X));
end

