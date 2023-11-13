clc
clear all
close all

rng('default');
data=readtable("housing.csv");
data(:,end)=[];
varNames=data.Properties.VariableNames;
data=table2array(data);

[m,n]=size(data);
p=0.75;
idx=randperm(m);
calibration_data=data(idx(1:round(p*m)),:);
test_data=data(idx(round(p*m)+1:end),:);

% Number of missing values for each variable
missing_values_cal=sum(ismissing(calibration_data));
missing_values_test = sum(ismissing(test_data));
fprintf('Missing values in the calibration data:\n');
disp(missing_values_cal);
fprintf('Missing values in the test data:\n');
disp(missing_values_test);

calibration_data = rmmissing(calibration_data);
test_data = rmmissing(test_data);

[calibration_data,mu,sigma]=zscore(calibration_data);
test_data=normalize(test_data,"Center",mu,"Scale",sigma);

x_train=calibration_data(:,1:end-1);
y_train=calibration_data(:,end);
y_calibration=y_train-mean(y_train);
x_calibration=x_train;

x_test=test_data(:,1:end-1);
y_test=test_data(:,end);
y_test=y_test-mean(y_test);

%% PCR
nsize=min(size(x_calibration));
[coeff,score,latent,tsquared,explained,mu] = pca(x_calibration,"Economy",false);
figure;
plot(1:nsize,100*cumsum(latent(1:nsize)/sum(latent(1:nsize))))
xlabel("Number of principal components")
ylabel("Explained Variance")

for i=1:nsize

    betapcs=regress(y_calibration,score(:,1:i));
    betavars_1=coeff(:,1:i)*betapcs;
    beta_pcr(:,i)=[mean(y_calibration)-mean(x_calibration)*betavars_1;betavars_1];

    n=length(y_train);
    yhat_pcr(:,i)=[ones(n,1) x_calibration]*beta_pcr(:,i);
    m=length(y_test);
    ypred_pcr(:,i)=[ones(m,1) x_test]*beta_pcr(:,i);

    Tss(i)=sum(y_calibration.^2);
    mse_pcr(i) = mean((y_test - ypred_pcr(:,i)).^2);


    Rss(i)=sum((y_train-yhat_pcr(:,i)).^2);
    R2_pcr(i)=1-Rss(i)/Tss(i);

    predRss(i)=sum((y_test-ypred_pcr(:,i)).^2);
    Q2_pcr(i)=1-predRss(i)/Tss(i);

end

%% PLS
for i = 1:min(size(x_calibration))
    [P,T,Q,U,beta_pls(:,i),var_pls, MSE,stats(:,i)]=plsregress(x_calibration,y_calibration,i,"cv",5);

    n = length(y_calibration);
    yhat_pls= [ones(n,1) x_calibration] * beta_pls(:,i);
    m=length(y_test);
    ypred_pls(:,i)=[ones(m,1) x_test] * beta_pls(:,i);

    TSS=sum((y_calibration-mean(y_calibration)).^2);
    % MSE
    mse_pls(i)=mean((y_test-ypred_pls(:,i)).^2);
    %R2
    RSS(i)=sum((y_calibration-yhat_pls).^2);
    R2_pls(i)=1 - RSS(i)/TSS;

    %Q2
    PRESS(i)=sum((y_test- ypred_pls(:,i)).^2);
    Q2_pls(i)=1-PRESS(i)/TSS;
end

%% Evaluate Models: Task 4, 5, 6

% Explained variance for PCR
cumulative_explained_X=cumsum(explained);
figure;
subplot(2,1,1);
plot(1:nsize,cumulative_explained_X(1:nsize),'b-o');
xlabel('Number of Principal Components');
ylabel('Cumulative Explained Variance in X');
title('Cumulative Explained Variance in X for PCR');

subplot(2,1,2);
plot(1:nsize, R2_pcr, 'r-o', 1:nsize, Q2_pcr,'g-o');
xlabel('Number of Principal Components');
ylabel('Explained Variance in Y');
legend('R^2','Q^2');
title('Explained Variance in Y (R^2 and Q^2) for PCR');

% Explained variance for PLSfigure;
figure;
plot(1:nsize,cumsum(var_pls(2,:)),'-bo');
hold on
plot(1:nsize,cumsum(var_pls(1,:)),'-ro');
xlabel('Number of PLS components');
ylabel('Percent Variance Explained in y');
legend(["Var Explained in Y", "Var Explained in X"]);
title('Cumulative Explained Variance in X and Y for PLS');


% Plot MSE and Q^2 against the number of latent variables
figure;
subplot(2, 1, 1);
plot(1:nsize, mse_pls, 'b-o');
xlabel('Number of Latent Variables (PLS)');
ylabel('MSE');
title('MSE Plot for PLS');

subplot(2, 1, 2);
plot(1:nsize, Q2_pls, 'r-o');
xlabel('Number of Latent Variables (PLS)');
ylabel('Q^2');
title('Q^2 Plot for PLS');

% Plot MSE and Q^2 against the number of principal components
figure;
subplot(2, 1, 1);
plot(1:nsize, mse_pcr, 'b-o');
xlabel('Number of Principal Components (PCR)');
ylabel('MSE');
title('MSE Plot for PCR');

subplot(2, 1, 2);
plot(1:nsize, Q2_pcr, 'r-o');
xlabel('Number of Principal Components (PCR)');
ylabel('Q^2');
title('Q^2 Plot for PCR');

% Variable importance in prediction
figure;
betas=[beta_pls(2:end,5),beta_pcr(2:end,6)];
bar(betas);
legend(["PLS Regression Coefficients", "PCR Regression Coefficients"]);
xticklabels(varNames);
title("Importance of Variables")

%% Recalibrate PLS
ind_pls=[1,2,5,6,7,8];
for i =1:6
    [Pn,Tn,Qn,Un,beta_plsn(:,i),var_plsn, MSE,statsn(:,i)]=plsregress(x_calibration(:,ind_pls),y_calibration,i,"cv",5);
    n = length(y_calibration);
    yhat_pls= [ones(n,1) x_calibration(:,ind_pls)]*beta_plsn(:,i);
    m=length(y_test);
    ypred_pls_new(:,i)=[ones(m,1) x_test(:,ind_pls)] *beta_plsn(:,i);

    TSS=sum((y_calibration - mean(y_calibration)).^2);
    % MSE
    New_mse_pls(i) = mean((y_test - ypred_pls_new(:,i)).^2);
    %R2
    RSS(i)=sum((y_calibration-yhat_pls).^2);
    New_R2_pls(i)=1 - RSS(i)/TSS;

    %Q2
    PRESS(i)=sum((y_test-ypred_pls_new(:,i)).^2);
    New_Q2_pls(i)=1-PRESS(i)/TSS;
end

%% Recalibrate PCR
ind_pcr=[1,2,4,5,6,7,8];
beta_pcr=[];
nsize=min(size(x_calibration));
[coeff,score,latent,tsquared,explained,mu] = pca(x_calibration(:,ind_pcr),"Economy",false);

for i=1:7

    betapcs=regress(y_calibration,score(:,1:i));
    betavars_1=coeff(:,1:i)*betapcs;
    beta_pcr(:,i)=[mean(y_train)-mean(x_calibration(:,ind_pcr))*betavars_1;betavars_1];

    n=length(y_train);
    yhat_pcr=[ones(n,1) x_calibration(:,ind_pcr)]*beta_pcr(:,i);
    m=length(y_test);
    ypred_pcr_new(:,i)=[ones(m,1) x_test(:,ind_pcr)]*beta_pcr(:,i);

    Tss(i)=sum(y_calibration.^2);
    New_mse_pcr(i) = mean((y_test -  ypred_pcr_new(:,i)).^2);


    Rss(i)=sum((y_train-yhat_pcr).^2);
    New_R2_pcr(i)=1-Rss(i)/Tss(i);

    predRss(i)=sum((y_test- ypred_pcr_new(:,i)).^2);
    New_Q2_pcr(i)=1-predRss(i)/Tss(i);

end
fprintf('Q2 after Recalibration for PCR: \n');
disp(New_Q2_pcr(end))
fprintf('Q2 after Recalibration for PLS: \n');
disp(New_Q2_pls(end))

%% Task 8
% True values and the predicted values against each other
figure;
subplot(1,2,1);
scatter(y_test,ypred_pcr_new(:,end),'filled');
hold on;
line([min(y_test),max(y_test)],[min(y_test),max(y_test)], 'Color', 'red', 'LineStyle', '--');
xlabel('True Values');
ylabel('Predicted Values');
title('True vs Predicted Values for PCR');
legend('Data Points', 'Perfect Prediction', 'Location', 'Northwest');
subplot(1,2,2);
scatter(y_test,ypred_pls_new(:,end),'filled');
hold on;
line([min(y_test),max(y_test)],[min(y_test),max(y_test)], 'Color', 'red', 'LineStyle', '--');
xlabel('True Values');
ylabel('Predicted Values');
title('True vs Predicted Values for PLS');
legend('Data Points', 'Perfect Prediction', 'Location', 'Northwest');


% Plot the residuals as a function of the true value.
figure;
residuals_pcr=y_test-ypred_pcr_new(:,end);
residuals_pls=y_test-ypred_pls_new(:,end);
subplot(1,2,1);
scatter(y_test,residuals_pls);
xlabel('True Values (y-test)');
ylabel('Residuals (PLS)');
title('Residuals vs True Values (PLS)');
subplot(1,2,2);
scatter(y_test, residuals_pcr);
xlabel('True Values (y-test)');
ylabel('Residuals (PCR)');
title('Residuals vs True Values (PCR)');