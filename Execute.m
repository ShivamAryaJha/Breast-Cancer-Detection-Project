%% Initialization
clear ; close all; clc

%% Load Data

data = csvread('BCP.txt');
X = data(:, 2: 10); y = data(:, 11);

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);
disp(size(X));
X = (featureNormalize(X));

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
theta = zeros(n + 1, 1);
initial_theta = theta;
% Compute and display initial cost and gradient

cost = costFunction(theta, X, y);
printf('Cost at initial theta (zeros): %f\n', cost);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%  This function will return theta and the cost 

  theta = gradientDescentMulti(X, y, theta, 0.001, 1000);
  %(exp(-X*theta))(1:100)
  (sigmoid(X*theta))(1:100)
  cost = costFunction(theta, X, y);

%  options = optimset('GradObj', 'on', 'MaxIter', 400);
%[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);
  

% Print theta to screen
fprintf('Cost at theta found by GradientDescent: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);


fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Part 4: Predict and Accuracies ==============

inp = input("Enter the attributes of the new tumor: ");
prob = sigmoid([1 inp] * theta);
fprintf(['\nFor a patient with given breast clump attributes,  ' ...
         'probability of it being malignant is %f\n'], prob);

% Compute accuracy on our training set
p = predict(theta, [1 inp]);

%accuracy
pos = find(y==2); neg = find(y == 4);
y(pos) = y(pos) - 2;
y(neg) = y(neg) - 3;

count=0;
P = predict(theta, X);
for i=1:m
  if(y(i,1)== P(i,1))
    count++;
    end
end

accuracy = count/m*100;
fprintf('\nTrain Accuracy: %f \n', accuracy);

% disp(y);
fprintf('\n');


