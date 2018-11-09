function [stateCollection, teacherCollection] = collectStateMatrix(trainInputSequence, trainOutputSequence, trained_esn, nForgetPoints)

% compute total size of sample points to be used
sampleSize = 0;
nTimeSeries = size(trainInputSequence, 1);
for i = 1:nTimeSeries
    sampleSize = sampleSize + size(trainInputSequence{i,1},1) - max([0, nForgetPoints]);
end

% collect input+reservoir states into stateCollection
stateCollection = zeros(sampleSize, trained_esn.nInputUnits + trained_esn.nInternalUnits);
collectIndex = 1;
for i = 1:nTimeSeries
    if strcmp(trained_esn.type, 'twi_esn')
        if size(trainInputSequence{i,1},2) > 1
            trained_esn.avDist = ...
                mean(sqrt(sum(((trainInputSequence{i,1}(2:end,:) - trainInputSequence{i,1}(1:end - 1,:))').^2)));
        else
            trained_esn.avDist = mean(abs(trainInputSequence{i,1}(2:end,:) - trainInputSequence{i,1}(1:end - 1,:)));
        end
    end           
    stateCollection_i = ...
        compute_statematrix(trainInputSequence{i,1}, trainOutputSequence{i,1}, trained_esn, nForgetPoints);
    l = size(stateCollection_i, 1);
    stateCollection(collectIndex:collectIndex+l-1, :) = stateCollection_i;
    collectIndex = collectIndex + l;
end

% collect teacher signals (including applying the inverse output
% activation function) into teacherCollection
teacherCollection = zeros(sampleSize, trained_esn.nOutputUnits);
collectIndex = 1;
for i = 1:nTimeSeries
    teacherCollection_i = ...
        compute_teacher(trainOutputSequence{i,1}, trained_esn, nForgetPoints);
    l = size(teacherCollection_i, 1);
    teacherCollection(collectIndex:collectIndex+l-1, :) = teacherCollection_i;
    collectIndex = collectIndex + l;
end


