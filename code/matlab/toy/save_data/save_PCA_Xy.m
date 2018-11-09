% PCA transformed data

clear; close all; 

% library
addpath(genpath('../../library'))

% what directory am I in?
datatask = 'toy';

% data directory
data_dir = ['../../../../data/' datatask];

% load data Xy
load([data_dir '/data_Xy.mat']); % data

% PCA
[~, score, latent] = pca(data.X);
pc.X = score;
pc.y = data.y;
pc.latent = latent;
pc.BLOCK_SIZE = data.BLOCK_SIZE;
pc.tag0 = data.tag0;
pc.tag1 = data.tag1;
pc.SUB_ID = data.SUB_ID;
pc.TEMP_ID = data.TEMP_ID;
save([data_dir '/data_PCA_Xy.mat'], 'pc');
