function [theta] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

pos = find(y==2); neg = find(y == 4);

y(pos) = y(pos) - 2;
y(neg) = y(neg) - 3;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %



theta = theta - alpha*(1/m)*X'*(sigmoid(X*theta) - y);

    % Save the cost J in every iteration    
   % J_history(iter) = costFunction(theta, X, y);

end

end
