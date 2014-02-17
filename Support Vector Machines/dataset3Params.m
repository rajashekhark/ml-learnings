function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
cVals = [0.01 0.03 0.1 0.3 1 3 10 30];	% initialize possible values for C
sigmaVals = [0.01 0.03 0.1 0.3 1 3 10 30]; % initialize possible values for sigma

% the intuition here is that by setting C, sigma and error to a very large value 
% we can avoid managing an array of errors for all the combinations of c and sigma
% instead we keep updating the parameters to the combination that generates least error
C = Inf;	
sigma = Inf;
error = Inf;
% iterate through the potential C and sigma values
for i = 1:length(cVals)
	currentCVal = cVals(i);
	for j = 1:length(sigmaVals)
		% train
		model= svmTrain(X, y, currentCVal, @(x1, x2) gaussianKernel(x1, x2, sigmaVals(j)));
		% predict
		predictions = svmPredict(model, Xval);
		% calculate error
		currentError = mean(double(predictions ~= yval));
		% update references if current error is lower than the previously recorded value
		if currentError < error,
			C = currentCVal;
			sigma = sigmaVals(j);
			error = currentError;
		end
	end
end

% =========================================================================

end
