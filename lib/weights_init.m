function W = rand_initialize(n_input_features, n_neurons_hidden_layer)
W = rand(n_neurons_hidden_layer, n_input_features+1);
return