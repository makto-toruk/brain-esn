% ESN config comparisons - classifier accuracy - (max +-1 avg) see paper
clear; close all;

% library
addpath(genpath('../library'))

datatask = 'toy'
TAU = [1, 2, 5, 10];
ALPHA = [0.2, 0.5, 0.8];

% load raw activation results
load(['../../../results/' datatask '/crossval_Xy_ith_.mat'])

% pick pos from train acc
acc = mean(train_acc);
loc = find(acc == max(acc));
pos = loc(end);
        
% apply on val
acc = mean(val_acc);
act_val = mean(acc(pos - 1:pos + 1));

y1 = []; y2 = [];
for ii = TAU
    for jj = ALPHA
        load(['../../../results/' datatask '/crossval_ESN_Xy_ith_' num2str(ii) '_' num2str(jj) '.mat'])
        
        % pick pos from train acc
        acc = mean(train_acc);
        loc = find(acc == max(acc));
        pos = loc(end);
        
        % corner cases
        if pos == 25
            pos = pos - 1;
        elseif pos == 1
            pos = pos + 1;
        end
        
        % apply on val
        acc = mean(val_acc);
        stdev = mean(std(val_acc(:, pos - 1:pos + 1)));
        len = size(val_acc, 1);
        
        y1 = [y1; mean(acc(pos - 1:pos + 1))];
        y2 = [y2; 1/sqrt(len)*stdev];
        
    end
end


% reshape for plotting
y1_test = y1(:, 1);
y2_test = y2(:, 1);
y1_test = reshape(y1_test, size(ALPHA, 2), size(TAU, 2))';
y2_test = reshape(y2_test, size(ALPHA, 2), size(TAU, 2))';

% val
figure;
h = barwitherr(1/sqrt(len)*y2_test, y1_test);
k = cell(1,4);
k{1} = '1'; k{2} = '2'; k{3} = '5'; k{4} = '10';
set(gca,'xticklabel', k)

hold on;

g=ishold(gca);
x=get(gca,'xlim');
line(x, [act_val act_val], 'Color', 'green');

l = cell(1,3);
l{1}='\alpha = 0.2'; l{2}='\alpha = 0.5'; l{3} = '\alpha = 0.8';   
legend(h,l);

title(['Validation accuracy for various leakage'])
ylabel('Max accuracy')
xlabel('\tau')

