function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	% vectorized implementation of simultaneously updating theta
	predictions = X * theta; % calculate predictions h(x) 
	errors = predictions - y; % calculate errors
	delta = (errors' * X)'; % statement below is the vectorized implementation of calculating the summation part for theta sum(i:1->n)(prediction) * x(i)  
	delta = delta / m; % divide by training set length
	delta = alpha .* delta; % multiply each element by alpha
		
	theta = theta - delta; % calculate theata (this is same as [theta = theta - (alpha * delta)])
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
	
	fprintf('Values for iteration %f \n', iter);
	fprintf('theta:\n');
	disp(theta);
	fprintf('cost:\n');
	disp(J_history(iter));

end

end
