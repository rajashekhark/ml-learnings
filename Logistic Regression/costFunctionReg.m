function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% calculate cost

[J, grad] = costFunction(theta, X, y);
regularizationValue = (lambda/(2 * m)) * sum(theta(2:end) .^ 2); % calculate the regularization value (lambda/2m) sum(theta(j) ^ 2) 
J = J + regularizationValue; % calculate the cost for normal logistic regression by calling the costFunction and then add regularization value to it

regularizeGrad = theta;	 % initialize the regularization vector to theta
regularizeGrad(1) = 0; % set the first element in regularized vector to zero. This would ensure that theta(0) is not affected
regularizeGrad = (lambda/m) .* regularizeGrad; % calculate regularization value for j > 1
grad = grad + regularizeGrad; % add regularization value to gradient. Note that the first element of the regularized vector is zero. So theta(0) will be unchanged.

% =============================================================

end
