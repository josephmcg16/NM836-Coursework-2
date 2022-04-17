function [phi, grad_phi] = sigmoid(z)
phi = 1./(1+exp(-z));
grad_phi = phi .* (1 - phi);
return