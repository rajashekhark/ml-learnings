function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% calculate cost
leniar_hypothesis = X * theta; % equivalent to (theat' * X)
g = sigmoid(leniar_hypothesis); % calculate the hypothesis value for logistic regression
onematrix = ones(size(g)); % initialize a ones matrix with size equal to the design matrix
positive = log(g); % start calculating the natural logarithm of the sigmoid 
positive = positive' * y; % calculate p(y=1) part of the cost function
positive = -1 .* positive; % negate the value
negative = log(onematrix - g); % start calculating y=0 part of the cost function
negative = negative' * (onematrix - y);  
J = sum(positive - negative)/m; % finally calculate the cost function

% calculate gradient
error = g - y;
grad = (error' * X)' % this should result in a n+1 * 1 vector
grad = grad / m;

% =============================================================

end
