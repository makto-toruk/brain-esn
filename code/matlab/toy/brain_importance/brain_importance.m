% correlate ROI ts with 'nDIM' top and bottom PCs
% aggregate importance across PCs

% keep 2*nROI at stage 1
% reduce to nROI at stage 2

% output:
% topROI: important ROI for task condition 1 (2b/ social/ scary)
% botROI: important ROI for task condition 2 (0b/ random/ funny)

clear;
close all;

datatask = 'toy';
data_dir = ['../../../../data/' datatask];

% parameters
nDIM = 5; % number of top and bottom PCs
TAU = 10; % TAU
ALPHA = 0.5; % ALPHA
nROI = 5; % number of ROI to display

% load data Xy
load([data_dir '/data_Xy.mat']); % data
BLOCK_SIZE = data.BLOCK_SIZE;

% load PC data
load([data_dir '/ESN/data_PCA_ESN_Xy_' num2str(TAU) '_' num2str(ALPHA) '.mat']);

% load weights
load(['../../../../results/' datatask '/wts_ESN_Xy_' num2str(TAU) '_' num2str(ALPHA) '.mat'])
[~, ord] = sort(weights);

% pc components
top_nDIM = fliplr(ord(end-(nDIM-1):end));
bot_nDIM = ord(1:nDIM);

wt_top = weights(top_nDIM);
wt_bot = weights(bot_nDIM);

% correlate over averages
task1 = data.X(data.y == 1, :);
task2 = data.X(data.y == 0, :);
for ii = 1:BLOCK_SIZE:size(task1,1)
    type1avg(:, :, (ii - 1)/BLOCK_SIZE + 1) = task1(ii: ii + BLOCK_SIZE - 1, :);
end
data1 = mean(type1avg, 3);
for ii = 1:BLOCK_SIZE:size(task2,1)
    type2avg(:, :, (ii - 1)/BLOCK_SIZE + 1) = task2(ii: ii + BLOCK_SIZE - 1, :);
end
data2 = mean(type2avg, 3);
data = [data1; data2];

clear type1avg type2avg

task1 = pc_ESN.X(pc_ESN.y == 1, top_nDIM);
task2 = pc_ESN.X(pc_ESN.y == 0, top_nDIM);
for ii = 1:BLOCK_SIZE:size(task1,1)
    type1avg(:, :, (ii - 1)/BLOCK_SIZE + 1) = task1(ii: ii + BLOCK_SIZE - 1, :);
end
top1 = mean(type1avg, 3);
for ii = 1:BLOCK_SIZE:size(task2,1)
    type2avg(:, :, (ii - 1)/BLOCK_SIZE + 1) = task2(ii: ii + BLOCK_SIZE - 1, :);
end
top2 = mean(type2avg, 3);
top = [top1; top2];

clear type1avg type2avg

task1 = pc_ESN.X(pc_ESN.y == 1, bot_nDIM);
task2 = pc_ESN.X(pc_ESN.y == 0, bot_nDIM);
for ii = 1:BLOCK_SIZE:size(task1,1)
    type1avg(:, :, (ii - 1)/BLOCK_SIZE + 1) = task1(ii: ii + BLOCK_SIZE - 1, :);
end
bot1 = mean(type1avg, 3);
for ii = 1:BLOCK_SIZE:size(task2,1)
    type2avg(:, :, (ii - 1)/BLOCK_SIZE + 1) = task2(ii: ii + BLOCK_SIZE - 1, :);
end
bot2 = mean(type2avg, 3);
bot = [bot1; bot2];

% correlate with original ts and multiply by wts to get "importance"
for i = 1:size(top, 2)
    for j = 1:size(data, 2)
        top_c(i,j) = corr(top1(:,i), data1(:,j)) * abs(wt_top(i));
        bot_c(i,j) = corr(bot2(:,i), data2(:,j)) * abs(wt_bot(i));
    end
end

% nROI original variables that best predict top and bot PCs
% threshold only 2*nROI of these

% find locations
for i = 1:size(top, 2)
    [~, ord] = sort(top_c(i,:), 'descend');
    top_thresh(i, :) = ord(1:nROI*2);
    [~, ord] = sort(bot_c(i,:), 'descend');
    bot_thresh(i, :) = ord(1:nROI*2);
end

% get importance values
topROI = zeros(size(top_c));
botROI = zeros(size(bot_c));
for i = 1:size(top, 2)
    topROI(nDIM - i + 1, top_thresh(i,:)) = top_c(i, top_thresh(i,:)); % top wts are at the end
    botROI(i, bot_thresh(i,:)) = bot_c(i, bot_thresh(i,:));
end

% aggregate information across PCs by adding their importance values

top_agg = sum(topROI);
bot_agg = sum(botROI);

% threshold to nROI values
[~, ord] = sort(top_agg, 'descend');
top_thresh = ord(1:nROI);
[~, ord] = sort(bot_agg, 'descend');
bot_thresh = ord(1:nROI);

% replace all but nROI with zeros
topROI = zeros(size(top_agg));
botROI = zeros(size(bot_agg));
topROI(top_thresh) = top_agg(top_thresh); 
botROI(bot_thresh) = bot_agg(bot_thresh);

% normalize these between 0 and 1 for display purposes
topROI = (topROI - min(topROI)) / ( max(topROI) - min(topROI) );
botROI = (botROI - min(botROI)) / ( max(botROI) - min(botROI) );
