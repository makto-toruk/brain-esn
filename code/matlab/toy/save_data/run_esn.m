% save ESN data

clear;
close all;

% save PCA transformed data
save_PCA_Xy;

% reservoir parameters
TAU = [1 2 5 10];
ALPHA = [0.2 0.5 0.8];

for ii = TAU
    for jj = ALPHA
    	% save reservoir outputs
        save_ESN_Xy(ii, jj);
        % save PCA transformed reservoir outputs
        save_PCA_ESN_Xy(ii, jj);
    end
end

% create folder for results
mkdir ../../../results/toy