%%%%%%% Time Series Forecasting Using Deep Learning  %%%%
%%%%%%% MATLAB 2022 b %%%%%%% Milad Moradi %%%%%%%%%%%%%%
clc
clear
close all

%% Step1: load Data
% number of numObservations= 1000 Sequnce in 3 channel
% Categaries: Sin, Square, Triangle, Sawtooth
load WaveformData  

numChannels = size(data{1},1);
figure
tiledlayout(2,2)   % similar to ((subPlot)) 
 for i = 4:7
nexttile
    % for plot of several variables with common x-axis
    stackedplot(data{i}','LineWidth',2,'DisplayLabels',"channel "+ (1:numChannels) )
     xlabel('Time Step')
 end

% 0.9 Data for Train , 0.1 Data for Test
numObservations = numel(data);
idxTrain = 1:floor(0.9*numObservations);
idxTest = floor(0.9*numObservations)+1:numObservations;
dataTrain = data(idxTrain);
dataTest = data(idxTest);

% or

cvp = cvpartition(numel(data),'Holdout',0.1);
dataTrain = data(training(cvp),:);
dataTest = data(test(cvp),:);

%% Step2: Prepare Data for Training

numTrain=numel(dataTrain);
XTrain=cell(1,numTrain);    
TTrain=cell(1,numTrain);    % TTrain=Target Train 
for n = 1:numTrain
    X = dataTrain{n};
    XTrain{n} = X(:,1:end-1);
    TTrain{n} = X(:,2:end);
end

% mean and unit variance 
muX = mean(cat(2,XTrain{:}),2);
sigmaX = std(cat(2,XTrain{:}),0,2);

muT = mean(cat(2,TTrain{:}),2);
sigmaT = std(cat(2,TTrain{:}),0,2);

for n = 1:numel(XTrain)
    XTrain{n} = (XTrain{n} - muX) ./ sigmaX;
    TTrain{n} = (TTrain{n} - muT) ./ sigmaT;
end

%% Step3: Define LSTM Network Architecture
layers = [
    sequenceInputLayer(numChannels)
    lstmLayer(128)
    fullyConnectedLayer(numChannels)
    regressionLayer];

% Training Options
options = trainingOptions("adam", ...
    MaxEpochs=200, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=0);
%% Step4: Train Neural Network
net = trainNetwork(XTrain,TTrain,layers,options);

%% Step5: Test Network

% Prepare TestData and normalize for Testing
numTest=numel(dataTest);
XTest=cell(1,numTest);    
TTest=cell(1,numTest);    % TTest=Target Test 

for n = 1:numTest
    X = dataTest{n};
    XTest{n} = (X(:,1:end-1) - muX) ./ sigmaX;
    TTest{n} = (X(:,2:end) - muT) ./ sigmaT;
end

YTest = predict(net,XTest,SequencePaddingDirection="left");

rmse=zeros(1,numTest);
for i = 1:size(YTest,1)
    rmse(i) = sqrt(mean((YTest{i} - TTest{i}).^2,"all"));
end

figure
 histogram(rmse)
  xlabel("RMSE")
  ylabel("Frequency")

meanrmse=mean(rmse);  % network accuracy  

%%  Step6: Open Loop Forecasting
% Forecast Future Time Steps
idx = 2;      % between 1 to 100
X = XTest{idx};
T = TTest{idx};

figure
 stackedplot(X','LineWidth',2,'DisplayLabels',"channel "+ (1:numChannels) )
  title("Test Observation " + idx)
  xlabel("Time Step")

net = resetState(net);
offset = 75;
[net,~] = predictAndUpdateState(net,X(:,1:offset));

numTimeSteps = size(X,2);
numPredictionTimeSteps = numTimeSteps - offset;
Y = zeros(numChannels,numPredictionTimeSteps);

for t = 1:numPredictionTimeSteps
    Xt = X(:,offset+t);
    [net,Y(:,t)] = predictAndUpdateState(net,Xt);
end

% Compare the predictions with the target values.
figure
 t = tiledlayout(numChannels,1);
  title(t,"Open Loop Forecasting")
 for i = 1:numChannels
     nexttile
     plot(T(i,:))
     hold on
     plot(offset:numTimeSteps,[T(i,offset) Y(i,:)],'--','LineWidth',2)
     ylabel("Channel " + i)
 end
  xlabel("Time Step")
 nexttile(1)
  legend(["Input" "Forecasted"])

%% Step7: Closed Loop Forecasting
net = resetState(net);
offset = size(X,2);
[net,Z] = predictAndUpdateState(net,X);

numPredictionTimeSteps = 200;
Xt = Z(:,end);
Y = zeros(numChannels,numPredictionTimeSteps);

for t = 1:numPredictionTimeSteps
    [net,Y(:,t)] = predictAndUpdateState(net,Xt);
    Xt = Y(:,t);
end

numTimeSteps = offset + numPredictionTimeSteps;

% Visualize the forecasted values in a plot.
figure
 t = tiledlayout(numChannels,1);
  title(t,"Closed Loop Forecasting")

 for i = 1:numChannels
     nexttile
     plot(T(i,1:offset))
     hold on
     plot(offset:numTimeSteps,[T(i,offset) Y(i,:)],'--','LineWidth',2)
     ylabel("Channel " + i)
 end
  xlabel("Time Step")
 nexttile(1)
  legend(["Input" "Forecasted"])
