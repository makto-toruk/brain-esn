function X_ESN = reservoir_outputs(X, BLOCK_SIZE, REM_STEP, LEAKAGE)

    addpath(genpath('../../../../library/ESNToolbox'));
    
    nInputUnits = size(X, 2); 
    nInternalUnits = REM_STEP*nInputUnits; 
    nOutputUnits = size(X, 2); % dummy
    
    X = mat2cell(X, BLOCK_SIZE*ones(size(X,1)/BLOCK_SIZE,1), nInputUnits);  

    %% generate an esn 

    inputScale = 0.1;
    inputShift = 0.1;
    outputScale = 0.3;
    outputShift = 0.1;
    leakage = LEAKAGE;

    % 
    esn = generate_esn(nInputUnits, nInternalUnits, nOutputUnits, ...
        'spectralRadius', 0.5, 'inputScaling', inputScale*ones(nInputUnits,1), 'inputShift', inputShift*ones(nInputUnits,1), ...
        'teacherScaling', outputScale*ones(nOutputUnits,1), 'teacherShift', outputShift*ones(nOutputUnits,1),'feedbackScaling', 0, ...
        'type', 'leaky_esn', 'leakage', leakage, 'learningMode', 'offline_multipleTimeSeries'); 

    esn.internalWeights = esn.spectralRadius * esn.internalWeights_UnitSR;

    %% collect all states
    nForgetPoints = 0; % discard the first N points

    % collect train state matrices and teachers
    [stateCollection, ~] = collectStateMatrix(X, X, esn, nForgetPoints); % second X is a dummy

    % no input
    stateCollection = stateCollection(:,1:end-nInputUnits);

    X_ESN = stateCollection;

end

