clc
clear all
close all

% Read the original image
Original_Image = imread('lui-peng-8NxTrV6i4WQ-unsplash.jpg');  

% Add salt & pepper noise to the original image
noisy_image = imnoise(Original_Image, 'salt & pepper',0.2);

% Read the disturbed image
disturbed_image = imread('disturbed.png');  

% Display the original, noisy and disturbed images
figure
subplot(3,1,1)
imshow(Original_Image);
title("Original Image")

subplot(3,1,2)
imshow(noisy_image);
title("Image with salt and pepper noise")

subplot(3,1,3)
imshow(disturbed_image);
title("Image with a shape on it")

% Convert the images to grayscale
Original_Gray = rgb2gray(Original_Image);
noisy_gray = rgb2gray(noisy_image);
disturbed_gray = rgb2gray(disturbed_image);

% Get the size of the grayscale images
[M, N] = size(Original_Gray); 

% Reshape the grayscale images into column vectors
X = double([reshape(Original_Gray, M * N, 1), reshape(noisy_gray, M * N, 1)]);
X2 = double([reshape(Original_Gray, M * N, 1), reshape(disturbed_gray, M * N, 1)]);

% Apply Robust Principal Component Analysis (RPCA)
[L, S] = RPCA(X);
[L2, S2] = RPCA(X2);

% Reshape the low rank and sparse matrices back into images
background_low_rank = uint8(reshape(L(:, 1), M, N));
object_sparse = uint8(reshape(S(:, 1), M, N));

background_low_rank2 = uint8(reshape(L2(:, 1), M, N));
object_sparse2 = uint8(reshape(S2(:, 1), M, N));

% Display the results
figure
subplot(2,1,1)
imshow(object_sparse);
title("Sparse Image for noisy_image")

subplot(2,1,2)
imshow(background_low_rank);
title("low rank Image for noisy_image")

figure
subplot(2,1,1)
imshow(object_sparse2);
title("Sparse Image for Image with a shape")

subplot(2,1,2)
imshow(background_low_rank2);
title("low rank Image for Image with a shape")

% RPCA function
function [L,S] = RPCA(X)
    [n1,n2] = size(X);
    mu = n1*n2/(4*sum(abs(X(:))));
    lambda = 1/sqrt(max(n1,n2));
    thresh = 1e-7*norm(X,'fro');
    L = zeros(size(X));
    S = zeros(size(X));
    Y = zeros(size(X));
    count = 0;
    while((norm(X-L-S,'fro')>thresh)&&(count<1000))
        L = SVT(X-S+(1/mu)*Y,1/mu);
        S = shrink(X-L+(1/mu)*Y,lambda/mu);
        Y = Y + mu*(X-L-S);
        count = count + 1;
        disp(count);
    end
end

% Singular Value Thresholding (SVT) function
function out = SVT(X, tau)
    [U, S, V] = svd(X, 'econ');
    out = U * shrink(S, tau) * V';
end

% Shrink function
function out = shrink(X,tau)
    out = sign(X).*max(abs(X)-tau,0);
end
