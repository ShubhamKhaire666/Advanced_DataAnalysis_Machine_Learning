clc
clear all;
close all;

% Load the data
DataFrame = readtable("housing.csv");

% Remove the "ocean_proximity" column
DataFrame(:,"ocean_proximity") = [];

% Separate the target variable (median_house_value)
YCalibration = DataFrame(:,"median_house_value");

% Remove the target variable from the DataFrame
DataFrame(:,"median_house_value") = [];

% Convert the DataFrame and target variable to arrays
XCalibration = table2array(DataFrame);
YCalibration = table2array(YCalibration);

% Check for missing values in XCalibration
sum(isnan(XCalibration))

% Check for missing values in YCalibration
sum(isnan(YCalibration))

% Remove rows with missing values in XCalibration
XCalibration = XCalibration(isnan(XCalibration(:,5)) == 0,:);
YCalibration = YCalibration(isnan(XCalibration(:,5)) == 0,:);

% Calculate the number of observations
ObservationNumbers = length(YCalibration);

% Create a partition for calibration and validation
PartitionPercentage = cvpartition(ObservationNumbers, 'HoldOut', 0.2);
idxCal = training(PartitionPercentage);
idxVal = test(PartitionPercentage);

% Separate the calibration and validation data
XCal = XCalibration(idxCal,:);
YCal = YCalibration(idxCal,:);

XVal = XCalibration(idxVal,:);
YVal = YCalibration(idxVal,:);

% Standardize the calibration data using z-score normalization
[XCal, XCalMean, XCalStandardDeviation] = zscore(XCal); 

% Normalize the validation data using the mean and standard deviation from the calibration data
XVal = normalize(XVal, 'Center', XCalMean, 'Scale', XCalStandardDeviation);

% Center the target variables
YCal = YCal - mean(YCal);
YVal = YVal - mean(YCal); 

% Perform Principal Component Analysis (PCA) on the calibration data
[coeff, Score, Latent, TSquared, ExplainedVariance] = pca(XCal,"Economy",false,"Centered",false);

% Perform Partial Least Squares Regression (PLSR) and Principal Component Regression (PCR) for different numbers of components
for i = 1:8
    % Perform PCR
    betaPCs = regress(YCal, Score(:,1:i));
    modelPCR(i).betaVars = coeff(:,1:i) * betaPCs;
    modelPCR(i).betaVars = [mean(YCal) - mean(XCal)*modelPCR(i).betaVars; modelPCR(i).betaVars]; 
    
    n = length(YCal);
    modelPCR(i).Yhat = [ones(n,1) XCal] * modelPCR(i).betaVars;
 
    m = length(YVal);
    modelPCR(i).YPred = [ones(m,1) XVal] * modelPCR(i).betaVars;

    modelPCR(i).Yhat = modelPCR(i).Yhat  + ones(n,1)*mean(YCalibration(idxCal));
    modelPCR(i).YPred = modelPCR(i).YPred + ones(m,1)*mean(YCalibration(idxCal));

    modelPCR(i).TSS = sum((YCalibration(idxCal) - mean(YCalibration(idxCal))).^2); 
    modelPCR(i).RSS = sum((YCalibration(idxCal) - modelPCR(i).Yhat).^2);
    modelPCR(i).R2 = 1 - modelPCR(i).RSS/modelPCR(i).TSS;
    modelPCR(i).PRESS = sum((YCalibration(idxVal) - modelPCR(i).YPred).^2);
    modelPCR(i).Q2 = 1 - modelPCR(i).PRESS/modelPCR(i).TSS;
end

% Perform PLSR for different numbers of components
for i = 1:8
    [modelPLS(i).P , modelPLS(i).T, modelPLS(i).Q, modelPLS(i).U, ...
        modelPLS(i).beta, modelPLS(i).var, modelPLS(i).MSE, modelPLS(i).stats] = plsregress(XCal, YCal, i, "cv",5);

    n = length(YCal);
    modelPLS(i).Yhat = [ones(n,1) XCal] * modelPLS(i).beta;
 
    m = length(YVal);
    modelPLS(i).YPred = [ones(m,1) XVal] * modelPLS(i).beta;

    modelPLS(i).TSS = sum((YCal - mean(YCal)).^2); 
    modelPLS(i).RSS = sum((YCal - modelPLS(i).Yhat).^2);
    modelPLS(i).R2 = 1 - modelPLS(i).RSS/modelPLS(i).TSS;
    modelPLS(i).PRESS = sum((YVal - modelPLS(i).YPred).^2);
    modelPLS(i).Q2 = 1 - modelPLS(i).PRESS/modelPLS(i).TSS;
end

% Plot the variance explained by the components
varNames = DataFrame.Properties.VariableNames(:);
figure;
plot(1:8,100 * cumsum(ExplainedVariance(1:8))/sum(ExplainedVariance(1:7)),'-g*');
hold on
plot(1:8,100*cumsum(modelPLS(8).var(2,:))/sum(modelPLS(8).var(2,:)),'-c*');
hold on
plot(1:8,100*cumsum(modelPLS(8).var(1,:))/sum(modelPLS(8).var(1,:)),'-k*');
xlabel('Components');
ylabel('Explained Variance');
legend(["Explained Variance PCA", "Explained Variance in Y for PLS", "Explained Variance in X for PLS"]);
axis tight
hold off;

