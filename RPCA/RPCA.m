clc
clear all
close all

% Load the image
img = imread('eye-mythological-dragon-fire-generative-ai (1).jpg');

% Convert to grayscale
img_gray = rgb2gray(img);

% Convert to double
img_double = im2double(img_gray);

% Apply RPCA
[L, S] = RPCA_function(img_double);

% Convert back to image for visualization
L_img = mat2gray(L);
S_img = mat2gray(S);

% Display the images
imshow(L_img);
figure;
imshow(S_img);

function [L,S] = RPCA_function(X)
[n1,n2] = size(X);
mu = n1*n2/(4*sum(abs(X(:))));
lambda = 1/sqrt(max(n1,n2));
thresh = 1e-7*norm(X,'fro');

S = zeros(size(X));
Y = zeros(size(X));
count = 0;
L = SVT(X-S+(1/mu)*Y,1/mu);
while((norm(X-L-S,'fro')>thresh)&&(count<1000))
    L = SVT(X-S+(1/mu)*Y,1/mu);
    S = shrink(X-L+(1/mu)*Y,lambda/mu);
    Y = Y+mu*(X-L-S);
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

