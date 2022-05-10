function [phi, phi_grad] = hyptan(z)
phi = (exp(2.*z) - 1) ./ (exp(2.*z) + 1);
phi_grad = 1 - phi.^2;
return