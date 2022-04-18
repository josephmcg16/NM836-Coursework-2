function phi_grad = relu_grad(z)
% matlab makes me do this since you can't index a function call?
phi = max(0, z);
phi_grad = phi./z;
return