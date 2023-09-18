clc
clear all
close all

%Load Data
wine_data = readtable("winequality-red.csv");

% Step 1: Visualize the dataset using boxplots
Column_Names = wine_data.Properties.VariableNames;

%Plot boxplot of the initital data
figure('Name','Boxplots')
boxplot(table2array(wine_data),'Orientation','vertical','Labels',Column_Names(1:end))
title("Box Plot of Wine Data");

% Step 2: Scale and center the dataset

% Normalize the dataset using the Z-score method
scaled_data = zscore(wine_data{:, 1:end});
normalized_wine_data = array2table(scaled_data, 'VariableNames', Column_Names(1:end));

% Create boxplots for the normalized data
figure('Name', 'Normalized Data Boxplots');
boxplot(table2array(normalized_wine_data), 'Orientation', 'vertical', 'Labels', Column_Names(1:end));
title('Boxplots for the scaled dataset');

% Step 3: Apply PCA
[loadings, scores, eigenvalues, T2_stats, explained_variance] = pca(scaled_data);

% Step 4: Visualize variation explained by different number of PCs
explained_variance
figure()
pareto(explained_variance)
xlabel('Principal Component')
ylabel('Explained Variance (%)')
title('Explained Variance by Principal Component');

% Step 5: Compute biplot of the first two principal components
figure
biplot(loadings(:, 1:2), 'Scores', scores(:, 1:2), 'Varlabels', Column_Names);
title('Biplot of the First Two Principal Components');


% Step 7: Create loading bar plots for the first principal component
figure('Name', 'Loading Coefficients for the First Component');
bar(loadings(:, 1)');
xticks(1:length(Column_Names)); % Set the x-axis tick positions
xticklabels(Column_Names);
xlabel('Variables');
ylabel('Loading Coefficients');
title('Loading Coefficients for the First Principal Component');

% Step 8: Plot T2 and SPE control charts (if applicable)
figure('Name', 'T2 Square Score');
T2_mean = mean(T2_stats);
T2_std = std(T2_stats);
T2_upper_limit = T2_mean + 3 * T2_std;
T2_lower_limit = T2_mean - 3 * T2_std;
sample_index = 1:length(T2_stats);
plot(sample_index, T2_stats), hold on
plot(sample_index, T2_upper_limit * ones(size(T2_stats)), 'b--')
xlabel('Sample');
ylabel('T2 Square Score');

