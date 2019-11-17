%% MLP - dataset: Fisheriris
%author = @leilamr

close all; 
clear all;
clc

%load dataset
load iris_dataset irisInputs irisTargets

%inputs data
inputs = irisInputs';
targets = irisTargets';

%% make network
% Mean squared error goal (default = 0.0)
goal = 0;
       
%Spread of radial basis functions (default = 1.0)
spread = 2.500;

%Maximum number of neurons
MN = 8;                                     
            
%Number of neurons to add between displays            
DF = 2;   
 
epochs = 100;
avg_acc = 0;
for i = 1:epochs
    %divide data into training and testing: 70% train, 30% test
    [m, n] = size(inputs);
    [o, p] = size(targets);

    P = 0.70;
    idx = randperm(m);

    trainInputs = inputs(idx(1:round(P*m)),:)'; 
    trainTargets = targets(idx(1:round(P*o)),:)';

    testInputs = inputs(idx(round(P*m)+1:end),:)';
    testTargets = targets(idx(round(P*o)+1:end),:)';

    %% train network
    net = newrb(trainInputs,trainTargets,goal,spread,MN,DF);

    %% test the Network

    testOutputs = net(testInputs);
    e = gsubtract(testTargets,testOutputs);

    performance = perform(net,testTargets,testOutputs);

    tind = vec2ind(testTargets);
    yind = vec2ind(testOutputs);
    percentErrors = sum(tind ~= yind)/numel(tind);

    acc = 100 * (1 - percentErrors);


    fprintf('Accuracy = %.3f%%\n', acc);
    
    avg_acc = avg_acc + acc;
end
avg_acc = avg_acc/epochs;
fprintf('Avg Accuracy = %.3f%%\n', avg_acc);
view(net)