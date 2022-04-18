function grad_phi = sigmoid_grad(z)
% matlab makes me do this since you can't index a function call?
phi = 1./(1+exp(-z));
grad_phi = phi .* (1 - phi);
return