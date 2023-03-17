% Big data small data Home taken assignment 1
% Concrete Compressive Strength 
%% Preparing workspace
clc
clear
close all

%% Loading dataset
load ConcreteDataHCT.mat

%Labelling the dataset (Header)
data = array2table(ConcreteDataHCT, 'VariableNames',{'cement',...
    'blast furnace slag','fly ash','water','superplasticizer',...
    'coarse aggregate','fine aggregate','age','concrete compressive strength'});

% Listing out summary of dataset 
summary(data)                       % missing data can be identified here  

% Assigning data to X and Y
x = table2array(data(:,1:8));       % Data type = Double
y = table2array(data(:,9));         % Data type = Double

%% Detecting  missing data (Verification)
TF_miss = ismissing(data);          % Checking the entire dataset
idx = find(TF_miss);                % index of the missing value
s = size(idx,1);                    % count of missing value  

%% Visualizing the dataset

% =================== Correlation ====================
figure('Name','Correlation','numbertitle','off')
R = corrplot(data,'testR','on','varNames',{'cement','blast furnace slag',...
    'fly ash','water','superplasticizer','coarse aggregate',...
    'fine aggregate','age','Concrete strength'});
fprintf('Correlation matrix ....\n')
display(R)

% =================== Box Plot ====================
[m,n] = size(x);
variablename = {'cement','blast furnace slag','fly ash','water',...
    'superplasticizer','coarse aggregate','fine aggregate','age'};
figure('Name','Box Plot','Numbertitle','off')
grid on;
for i =1:n
    subplot(4,2,i)
    boxplot(x(1:m,i),'orientation','horizontal')
    title(sprintf(string(variablename(1,i))))
    set(gca,'FontSize',10)
    set(gca,'xticklabel',[])
end
sgtitle("Features",'FontSize',10);

% =================== Scatter Plot ====================
figure('Name','Scatter plot 1')
title('CC Strength vs (Cement, Age, Water)')
hold on; grid on;
g = [x(:,4),x(:,8)];
gscatter(x(:,1),y,g,'br','ox')
xlabel('cement')
ylabel('compressive concrete strength')
legend('water','age')
hold off;

figure('Name','Scatter plot 2')
title('CC Strength vs (Fly Ash, Super Plasticizer, Fine aggregate)')
hold on; grid on;
g = [x(:,3),x(:,5),x(:,7)];
gscatter(x(:,1),y,g,'brm','ox.')
xlabel('cement')
ylabel('compressive concrete strength')
legend('Fly ash','Super Plasticizer', 'Fine aggregate')
hold off;

%% Count and Remove outlier 

fprintf('Detecting and removing outlier using quartile method....\n')
[X_c, TF]= rmoutliers(x,'quartile');
ind = find(TF);         % Index of the outliers
s_o = size(ind,1);      % No. of outliers

% Removing the element in y with same index number
y_c = y;
y_c(ind) = [];

%% Normalization
% Feature scaling 
[X, mu, sigma] = normalize(X_c);
X = [ones(size(X,1),1) X];

%% Randomizing and dividing the dataset 
rng(2); 
[idx_train,idx_val,idx_test] = dividerand(height(X),0.6,0.20,0.20);

train = idx_train'; % random index for train
val = idx_val';     % random index for cross validation
test = idx_test';   % random index for test

%% Assigning divided dataset 
% Train dataset as 60 percent
X_train = X(train,:);
Y_train = y_c(train,:);

% Cross validation dataset as 20 percent
X_cv = X(val,:);
Y_cv = y_c(val,:);

% Test dataset as 20 percent 
X_test = X(test,:);
Y_test = y_c(test,:);

%% Gradient descent without regularization

% ========== Finding theta and cost for training set ==========
figure('Name','Training error','Numbertitle','off')
title('Cost based on learning rate')
hold on;
for j = 1:6    
    alpha = [0.001 0.005 0.01 0.05 0.1 0.5];      % Learning rate 
    iterations = 2000;           % No. of iterations 
    lambda_without = 0;                  % Regularization parameter 
    para = size(X_train,2);
    int_theta = zeros(para,1); % initializing the theta for each feature 

    [theta, J_cost_train] = graidentDescent(X_train, Y_train, int_theta, alpha(j), iterations, lambda_without);
   
    % J_cost vs number of iterations 
    txt1 = ['Alpha = ',num2str(alpha(j))];
    plot(1:numel(J_cost_train), J_cost_train, 'LineWidth', 2,'DisplayName',txt1);
    xlabel('No. of iterations');
    ylabel('J cost');
end
grid on;hold off;
legend show 

%% Predict and evaluate (Training dataset)
fprintf('Prediction 10 random training samples...\n')
observed_train= X_train * theta;
measured_train = Y_train;
for i = 1:10
  sample = randi(length(X_train));
  predict = observed_train(sample, 1);
  actual = measured_train(sample, 1);
  fprintf('Sample: %d , predicted: %f, actual: %f\n', sample, predict, actual);  
end

RMSE_train = sqrt(mean((observed_train - measured_train).^2))/100

%% Gradient descent with regularization

min_cost_value =zeros(11,1);
J_cost_cv = zeros(11,1);

for j = 1:11    
    % ========== Finding theta and cost for training set ==========
    alpha = 0.5;                        % Learning rate 
    iterations = 2000;                   % No. of iterations 
    lambda = (0:0.1:1);                  % Regularization parameter 
    para = size(X_train,2);
    int_theta = zeros(para,1);            % initializing the theta for each feature 

    [theta_lamb, J_cost_train] = graidentDescent(X_train, Y_train, int_theta, alpha, iterations, lambda(j));
    min_cost_value(j,1)= min(J_cost_train);
    
    % ========== Finding cost for Cross validation set ==========
    J_cost_cv(j,1) = computecost(X_cv,Y_cv,theta,lambda(j));
end
 % ========== Plotting lambda vs error ==========
figure('Name','Choosing the regularization parameter','Numbertitle','off')
title('Cost based on regularization')
plot(lambda,min_cost_value,'b')
hold on;
plot(lambda,J_cost_cv,'r')
ylim([20 50])
xlabel('Lambda')
ylabel('Error')
grid on;hold off;
legend('J train','J cv')

% Choosing lambda as 0.5
alpha = 0.5;                        % Learning rate 
iterations = 2000;                   % No. of iterations 
lambda = 0.5;                  % Regularization parameter 
para = size(X_train,2);
int_theta = zeros(para,1);            % initializing the theta for each feature 

[theta_lamb_opt, J_cost_train] = graidentDescent(X_train, Y_train, int_theta, alpha, iterations, lambda);

%% Predict and evaluate cross validation (Trained with regularization)
observed_cv_opt= X_cv* theta_lamb_opt;
measured_cv_opt = Y_cv;
for i = 1:10
  sample = randi(length(X_cv));
  predict = observed_cv_opt(sample, 1);
  actual = measured_cv_opt(sample, 1);
  fprintf('Sample: %d , predicted: %f, actual: %f\n', sample, predict, actual);  
end

RMSE_cv_opt = sqrt(mean((observed_cv_opt - measured_cv_opt).^2))/100

%% Evaluation with test set
observed_test= X_test * theta_lamb_opt;
measured_test = Y_test;
for i = 1:10
  sample = randi(length(X_test));
  predict = observed_test(sample, 1);
  actual = measured_test(sample, 1);
  fprintf('Sample: %d , predicted: %f, actual: %f\n', sample, predict, actual);  
end

RMSE_test = sqrt(mean((observed_test - measured_test).^2))/100