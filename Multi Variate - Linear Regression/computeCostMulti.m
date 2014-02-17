function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.




% The implementation below to compute cost is equivalent to the commented code. 
% Trying this implementation to follow the instructions in the exercise. 
%*******predictions = X * theta;  % this is equivalent to the notation (theta' * X) as X has examples in its rows
%*******sqrErrors = (predictions-y).^2; % calculate square errors 
%*******J = 1/(2 * m) * sum(sqrErrors); % calculate cost
predictions = X * theta; 
errors = predictions - y;
J = 1/(2 * m) * (errors' * errors);

% =========================================================================

end
