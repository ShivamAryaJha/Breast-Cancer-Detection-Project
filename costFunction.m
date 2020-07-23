function [J, grad] = costFunction(theta, X, y)


% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

pos = find(y==2); neg = find(y == 4);
y(pos) = y(pos) - 2;
y(neg) = y(neg) - 3;
% ====================== YOUR CODE HERE ======================


J = (1/m)*sum(- y.*log(sigmoid(X*theta)) - (1 - y).*log(1 - sigmoid(X*theta)));
grad = (1/m)*X'*(sigmoid(X*theta) - y);

% =============================================================

end
