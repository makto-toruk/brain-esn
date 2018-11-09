% save ESN data

clear;
close all;

save_PCA_Xy;

TAU = [1 2 5 10];
ALPHA = [0.2 0.5 0.8];

for ii = TAU
    for jj = ALPHA
        save_ESN_Xy(ii, jj);
        save_PCA_ESN_Xy(ii, jj);
    end
end
