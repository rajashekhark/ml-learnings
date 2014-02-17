function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g = exp(-1.* z); % calculate exponent function value for every matrix element e(-z)
g = ones(size(z)) + g; % create a ones matrix and add it to the result above (this will calculate 1 + e(-z))
g = 1 ./ g; % take the inverse of every matrix element to get 1/1+e(-z) - sigmoid function

% =============================================================

end
