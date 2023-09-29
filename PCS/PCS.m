clc 
clear all
close all


% load the CSV file into a table
data = readtable("housing.csv");

% Remove Ocean Proximity Column
data(:,10)=[];
data(:,5) = [];

% Find rows with missing values
rows_with_missing_values = any(ismissing(data), 2);

% Display rows with missing values
disp(data(rows_with_missing_values, :));

% Remove rows with missing values from the original data
data(rows_with_missing_values, :) = [];

% Create an index for the calibration and test partitions
rng('default'); % Set the random seed for reproducibility
n = size(data, 1); % Size of dataset
calibrationFraction = 0.8; % Percentage of data for calibration
partitionIdx = randperm(n);
calibrationIdx = partitionIdx(1:round(calibrationFraction * n));
testIdx = partitionIdx(round(calibrationFraction * n) + 1:end);

% Separate the data into calibration and test sets
calibrationData = data(calibrationIdx, :);
testData = data(testIdx, :);

% Calculate the mean and standard deviation of the calibration partition
calibrationMean = mean(calibrationData{:,:});
calibrationStd = std(calibrationData{:,:});

% Use the mean and standard deviation of the calibration partition to center the test data
testDataCentered = (testData{:,:} - calibrationMean) / calibrationStd;

% Extract the numeric data from the table (assuming all columns are numeric)
calibrationDataNumeric = table2array(calibrationData);

% Perform PCA on the centered and scaled calibration data
[coeff,score,latent,tsquared,explained,mu] = pca(calibrationDataNumeric,"Economy",false);

pareto(explained);

% Choose the number of principal components (latent variables) to use
numComponents = 5; % You can adjust this number based on your analysis

% Select the first 'numComponents' principal components
selectedComponents = score(:, 1:numComponents);

% Perform multiple linear regression (PCR) with the selected components
b = regress(YCalibration, [ones(size(selectedComponents, 1), 1), selectedComponents]);

% The vector 'b' contains the regression coefficients for PCR

