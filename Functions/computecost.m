function J = computecost(X,y,theta,lambda)
m = length(y);
J = 0; %initialize cost
hypothesis = X*theta;
J = (1/(2*m)) * sum((hypothesis-y).^2) + (lambda/(2 * m)) * sum(theta(2:end).'*(theta(2:end)));
end