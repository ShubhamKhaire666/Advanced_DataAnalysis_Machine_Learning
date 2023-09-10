% Task 1: Load and Display Grayscale Image
originalImage = imread('DragonEye.jpg');
grayImage = rgb2gray(originalImage);

% Display the grayscale image
subplot(2, 2, 1);
imshow(grayImage);
title('Original Grayscale Image');

% Task 2: Add Three Types of Noise
noisyImage1 = imnoise(grayImage, 'gaussian', 0, 0.01); % Gaussian noise
noisyImage2 = imnoise(grayImage, 'salt & pepper', 0.05); % Salt and Pepper noise
noisyImage3 = imnoise(grayImage, 'speckle', 0.04); % Speckle noise

% Display the noisy images
subplot(2, 2, 2);
imshow(noisyImage1);
title('Noisy Image (Gaussian)');
subplot(2, 2, 3);
imshow(noisyImage2);
title('Noisy Image (Salt & Pepper)');
subplot(2, 2, 4);
imshow(noisyImage3);
title('Noisy Image (Speckle)');

% Task 3: Singular Value Decomposition and Cumulative Singular Values
[U1, S1, V1] = svd(double(noisyImage1));
[U2, S2, V2] = svd(double(noisyImage2));
[U3, S3, V3] = svd(double(noisyImage3));


% Calculate cumulative singular values
cumulativeS1 = cumsum(diag(S1)) / sum(diag(S1));
cumulativeS2 = cumsum(diag(S2)) / sum(diag(S2));
cumulativeS3 = cumsum(diag(S3)) / sum(diag(S3));

% Plot cumulative singular values
figure;
subplot(3, 1, 1);
plot(cumulativeS1);
title('Cumulative Singular Values (Gaussian)');
subplot(3, 1, 2);
plot(cumulativeS2);
title('Cumulative Singular Values (Salt & Pepper)');
subplot(3, 1, 3);
plot(cumulativeS3);
title('Cumulative Singular Values (Speckle)');

% Task 4: Image Reconstruction and RMSE Calculation
numSingularValues = 1:min(size(S1));
rmse1 = zeros(size(numSingularValues));
rmse2 = zeros(size(numSingularValues));
rmse3 = zeros(size(numSingularValues));

for i = numSingularValues
    % Reconstruct images using i singular values
    reconstructedImage1 = U1(:, 1:i) * S1(1:i, 1:i) * V1(:, 1:i)';
    reconstructedImage2 = U2(:, 1:i) * S2(1:i, 1:i) * V2(:, 1:i)';
    reconstructedImage3 = U3(:, 1:i) * S3(1:i, 1:i) * V3(:, 1:i)';
    
    % Calculate RMSE
    rmse1(i) = sqrt(mean((double(grayImage(:)) - double(reconstructedImage1(:))).^2));
    rmse2(i) = sqrt(mean((double(grayImage(:)) - double(reconstructedImage2(:))).^2));
    rmse3(i) = sqrt(mean((double(grayImage(:)) - double(reconstructedImage3(:))).^2));
end

% Plot RMSE values
figure;
plot(numSingularValues, rmse1, 'r', 'DisplayName', 'Gaussian');
hold on;
plot(numSingularValues, rmse2, 'g', 'DisplayName', 'Salt & Pepper');
plot(numSingularValues, rmse3, 'b', 'DisplayName', 'Speckle');
xlabel('Number of Singular Values');
ylabel('RMSE');
legend('Location', 'Best');
title('RMSE vs. Number of Singular Values');

% Task 5: Optimal Number of Singular Values
% Analyze the RMSE plot to identify the optimal values.
optimalSingularValues1 = find(rmse1 == min(rmse1)); 
optimalSingularValues2 = find(rmse2 == min(rmse2)); 
optimalSingularValues3 = find(rmse3 == min(rmse3));  

% Display the optimal values
fprintf('Optimal Singular Values for Gaussian Noise: %d\n', optimalSingularValues1);
fprintf('Optimal Singular Values for Salt & Pepper Noise: %d\n', optimalSingularValues2);
fprintf('Optimal Singular Values for Speckle Noise: %d\n', optimalSingularValues3);


% Task 6: Image Compression
% Calculate the compression ratio for each noisy image using the optimal number of singular values.

[rows cols] = size(double(grayImage));
CompressionRatio = 100 * [optimalSingularValues1 optimalSingularValues2 optimalSingularValues3] * (rows + cols)/(rows * cols)


% Display compression ratios
fprintf('Compression Ratio for Gaussian Noise: %.2f\n', CompressionRatio(1));
fprintf('Compression Ratio for Salt & Pepper Noise: %.2f\n', CompressionRatio(2));
fprintf('Compression Ratio for Speckle Noise: %.2f\n', CompressionRatio(3));

