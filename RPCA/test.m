clc
clear all

Original_Image = imread('eye-mythological-dragon-fire-generative-ai (1).jpg');  
noisy_image = imnoise(Original_Image, 'salt & pepper',0.2);
disturbed_image = imread('disturbed.png');  

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


Original_Gray = rgb2gray(Original_Image);
noisy_gray = rgb2gray(noisy_image);
disturbed_gray = rgb2gray(disturbed_image);

[M, N] = size(Original_Gray);  % Assuming both images have the same dimensions
X = double([reshape(Original_Gray, M * N, 1), reshape(noisy_gray, M * N, 1)]);

X2 = double([reshape(Original_Gray, M * N, 1), reshape(disturbed_gray, M * N, 1)]);


%% Apply RPCA
[L, S] = RPCA(X);
[L2, S2] = RPCA(X2);

%% Display results
background_low_rank = reshape(L(:, 1), M, N);
object_sparse = reshape(S(:, 1), M, N);

background_low_rank2 = reshape(L2(:, 1), M, N);
object_sparse2 = reshape(S2(:, 1), M, N);


figure
subplot(2,1,1)
imshow(uint8(object_sparse));
title("Sparse Image for noisy_image")
subplot(2,1,2)
imshow(uint8(background_low_rank));
title("low rank Image for noisy_image")

figure
subplot(2,1,1)
imshow(uint8(object_sparse2));
title("Sparse Image for Image with a shape")
subplot(2,1,2)
imshow(uint8(background_low_rank2));
title("low rank Image for Image with a shape")


% Functions taken from Chapter 3 of data driven science and engineering
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
end
end

function out = SVT(X, tau)
    [U, S, V] = svd(X, 'econ');
    out = U * shrink(S, tau) * V';
end

function out = shrink(X,tau)
    out = sign(X).*max(abs(X)-tau,0);
end