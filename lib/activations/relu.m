function [phi, phi_grad] = relu(z)
phi = max(0, z);
phi_grad = phi./z;
return