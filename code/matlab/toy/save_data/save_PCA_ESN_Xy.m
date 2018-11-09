% save reservoir outputs for analysis in python

function save_PCA_ESN_Xy(TAU, ALPHA)

% library
addpath(genpath('../../library'))

% what directory am I in?
datatask = 'toy';

data_dir = ['../../../../data/' datatask '/ESN'];

% load data_ESN
load([data_dir '/data_ESN_Xy_' num2str(TAU) '_' num2str(ALPHA) '.mat']); % data_ESN

% PCA
[~, score, latent] = pca(data_ESN.X);
pc_ESN.X = score;
pc_ESN.latent = latent;
pc_ESN.y = data_ESN.y;
pc_ESN.BLOCK_SIZE = data_ESN.BLOCK_SIZE;
pc_ESN.TAU = TAU;
pc_ESN.ALPHA = ALPHA;
pc_ESN.tag0 = data_ESN.tag0;
pc_ESN.tag1 = data_ESN.tag1;
pc_ESN.SUB_ID = data_ESN.SUB_ID;
pc_ESN.TEMP_ID = data_ESN.TEMP_ID;

save([data_dir '/data_PCA_ESN_Xy_' num2str(TAU) '_' num2str(ALPHA) '.mat'], 'pc_ESN');