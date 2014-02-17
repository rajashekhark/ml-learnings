function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

leniar_hypothesis = X * theta; % equivalent to (theat' * X)

% calculate sigmoid
g = exp(-1.* leniar_hypothesis); % calculate exponent function value for every matrix element e(-z)
g = ones(size(leniar_hypothesis)) + g; % create a ones matrix and add it to the result above (this will calculate 1 + e(-z))
g = 1 ./ g; % take the inverse of every matrix element to get 1/1+e(-z) - sigmoid function

% calculate unregularized cost function
onematrix = ones(size(g)); % initialize a ones matrix with size equal to the design matrix
positive = log(g); % start calculating the natural logarithm of the sigmoid 
positive = positive' * y; % calculate p(y=1) part of the cost function
positive = -1 .* positive; % negate the value
negative = log(onematrix - g); % start calculating y=0 part of the cost function
negative = negative' * (onematrix - y);  
J = sum(positive - negative)/m; % finally calculate the un regularized cost function

% regularize cost function
regularizationValue = (lambda/(2 * m)) * sum(theta(2:end) .^ 2); % calculate the regularization value (lambda/2m) sum(theta(j) ^ 2) 
J = J + regularizationValue; % calculate the cost for normal logistic regression by calling the costFunction and then add regularization value to it

% calculate unregularized gradient
beta = g - y;
grad = X' * beta; % this should result in a n+1 * 1 vector
grad = grad / m;

% regularize gradient
regularizeGrad = theta;	 % initialize the regularization vector to theta
regularizeGrad(1) = 0; % set the first element in regularized vector to zero. This would ensure that theta(0) is not affected
regularizeGrad = (lambda/m) .* regularizeGrad; % calculate regularization value for j > 1
grad = grad + regularizeGrad; % add regularization value to gradient. Note that the first element of the regularized vector is zero. So theta(0) will be unchanged.

% =============================================================

grad = grad(:);

end
