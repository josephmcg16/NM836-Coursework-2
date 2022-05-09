function W = weights_init(n_prev, n_next, method)
%WEIGHTS_INIT

if method == "xaviar"
    U = rand(n_next, n_prev+1);  % uniform dist
    W = normalize(U, "range",  [-1/sqrt(n_prev+1), 1/sqrt(n_next)]);

elseif method == "uniform"
    U = rand(n_next, n_prev+1);  % uniform dist
    W = normalize(U, "range",  [-1, 1]);
end
return