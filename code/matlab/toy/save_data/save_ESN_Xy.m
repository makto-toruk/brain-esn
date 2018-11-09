% save reservoir outputs for analysis in python

function save_ESN_Xy(TAU, ALPHA)

% change seed if it throws warnings
rng('default');
rng(42);

% library
addpath(genpath('../../library'))

% what directory am I in?
datatask = 'toy';

% make directory for saving
save_dir = ['../../../../data/' datatask '/ESN'];
if ~exist(save_dir, 'dir')
    mkdir(save_dir)
end

% load data Xy
load(['../../../../data/' datatask '/data_Xy.mat']); % data

X = data.X;
BLOCK_SIZE = data.BLOCK_SIZE;

% save reservoir outputs
X_ESN = reservoir_outputs(X, BLOCK_SIZE, TAU, ALPHA);

data_ESN.X = X_ESN;
data_ESN.y = data.y;
data_ESN.BLOCK_SIZE = data.BLOCK_SIZE;
data_ESN.TAU = TAU;
data_ESN.ALPHA = ALPHA;
data_ESN.tag0 = data.tag0;
data_ESN.tag1 = data.tag1;
data_ESN.SUB_ID = data.SUB_ID;
data_ESN.TEMP_ID = data.TEMP_ID;

save([save_dir '/data_ESN_Xy_' num2str(TAU) '_' num2str(ALPHA) '.mat'], 'data_ESN');