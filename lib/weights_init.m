function W = weights_init(n_previous_layer, n_next_layers)
W = rand(n_next_layers, n_previous_layer+1) ... 
    .* sqrt(1 / n_previous_layer);
return