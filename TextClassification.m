%%%%%%% Text Classification Using Deep Learning  %%%%
%%%%%%% MATLAB 2022-b %%%%%%% Milad Moradi %%%%%%%%%%
clc
clear
close all

%% Step1: Import data

data = readtable("factoryReports.csv"); 

data.Category = categorical(data.Category);
figure
 histogram(data.Category);
  xlabel("Class")
  ylabel("Frequency")
  title("Class Distribution")

% 0.8 data for Train and 0.2 data for test
cvp = cvpartition(data.Category,'Holdout',0.2);
dataTrain = data(training(cvp),:);
dataTestation = data(test(cvp),:);

textDataTrain = dataTrain.Description;
textDataTestation = dataTestation.Description;
YTrain = dataTrain.Category;     % Training class
YTest = dataTestation.Category;  % Test class

figure
 wordcloud(textDataTrain);
  title("Training Data Cloud")

%% Step2: preprocess the Textdata

documentsTrain = PreprocessTextF(textDataTrain);
documentsTrain(1:5)

%% setp3: Convert the words to numeric sequences

% Word encoding model to map words to indices and back
enc = wordEncoding(documentsTrain); 
documentLengths = doclength(documentsTrain);
figure
 histogram(documentLengths)
  title("Document Lengths")
  xlabel("Length")
  ylabel("Number of Documents")

% Most of the training documents have fewer than 7 tokens...
sequenceLength = 6;
XTrain = doc2sequence(enc,documentsTrain,'Length',sequenceLength);
XTrain(1:5)

%% step4: Create LSTM Network

inputSize = 1;
embeddingDimension = 50;
numHiddenUnits = 80;

numWords = enc.NumWords;
numClasses = numel(categories(YTrain));

layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

% training options
miniBatchSize = 16;
MaxEpoch = 50;
InitialLearnRate = 0.001;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpoch',MaxEpoch,...
    'InitialLearnRate', InitialLearnRate,...
    'LearnRateSchedule', 'piecewise',...
    'LearnRateDropFactor', 0.9000,...
    'LearnRateDropPeriod', 15,...
    'ExecutionEnvironment','gpu', ...
    'L2Regularization', 1.0000e-04,...
    'GradientThreshold',2, ...
    'Shuffle','never', ...
    'Plots','training-progress', ...
    'Verbose',false);

%% Step5: Train LSTM Network

[net,info] = trainNetwork(XTrain,YTrain,layers,options);

%% Step6: Test Network

documentsTest = PreprocessTextF(textDataTestation);
Xtest = doc2sequence(enc,documentsTest,'Length',sequenceLength);

YPred = classify(net,Xtest);
acc = sum(YPred == YTest)./numel(YTest);

%% Step7: Predict Using New Data

% new reports
NewText = [ ...
    "Coolant is pooling underneath sorter."
    "Sorter blows fuses at start up."
    "There are some very loud rattling sounds coming from the assembler."];


NewText = [ ...
    "میوه موز چند قیمت است؟."
     "موتور پراید کم مصرف است"
     "پراید ارزان قیمت است"]; 
 

% preprocess the NewText such as training documents
Newdocuments = PreprocessTextF(NewText);
Newdocuments(1:3) 

% Convert the NewText to sequences such as training sequences
XNew = doc2sequence(enc,Newdocuments,'Length',sequenceLength);

% Classify 
labelsNew = classify(net,XNew);