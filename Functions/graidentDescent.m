function [theta,J_history] = graidentDescent(X,y,theta,alpha,iterations,lambda)
    m = length(y);
    for i =1:iterations
        hypothesis = X*theta; % hypothesis vector
        % term1 = (alpha/m) * ((hv-y)' * X)';
        % term2= (lambda/m) * theta;
        theta = theta*(1-(lambda*alpha)/m) - (alpha/m)*sum((hypothesis - y).*X)';
        % theta= theta - term1
        % -term2;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        J_history(i) = computecost(X,y,theta,lambda); % calculating cost function
    end
end



