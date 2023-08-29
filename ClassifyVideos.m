%%%%%%% Classify Videos Using Deep Learning  %%%%%%%%
%%%%%%% MATLAB 2022-b %%%%%%% Milad Moradi %%%%%%%%%%
clc
clear
close all

dolstm=0; % 1 for Trian LSTM, 0 for load LSTM

%% Step 1: Load Pretrained Convolutional Network
netCNN =googlenet; % 'inceptionv3' 'googlenet'

%% Step 2: Load Data

% The Video of dataset format is .avi
% Size of dataset is 2 GB 
% contains 6763 clips over 51 classes such as "drink", "run"
% Width : 320 pixels and Height 240 pixels
% Viedoos resolution: 320*240

% dataset for this code:
% contains 593 clips over 4 classes: "run", "sit", "talk", "smile"
dataFolder = "hmdb51_org"; 
fraction = 1;
[files,labels] = hmdb51Files(dataFolder,fraction);

% Read the first video
idx = 1;
filename = files(idx);
video = readVideo(filename);
size(video)
labels(idx)

% view the video
numFrames = size(video,4);
figure
 for i = 1:numFrames
     frame = video(:,:,:,i);
     imshow(frame/255);
     drawnow %for updates figures
 end

%% Step 3: Convert Frames to Feature Vectors

% To read the video data and resize it...
% ... to match the input size of the GoogLeNet network

inputSize = netCNN.Layers(1).InputSize(1:2);  % CNN resolution: 240*240
layerName = "pool5-7x7_s1";  % last pooling layer

sequencesFile = fullfile(cd,"hmdb51_org_sequences.mat"); % cd: current directory

if exist(sequencesFile,'file') % Checks only for files or folders.
    load(sequencesFile,"sequences")
else
    numFiles = numel(files);
    sequences = cell(numFiles,1);
    
    for i = 1:numFiles
        fprintf("Reading file %d of %d...\n", i, numFiles)
        
        video = readVideo(files(i));
        video = centerCrop(video,inputSize);
        
        sequences{i,1} = activations(netCNN,video,layerName,'OutputAs','columns');
    end

    save(sequencesFile,"sequences");
    % or
    % save(sequencesFile,"sequences",v7.3);
    % v7.3 for: ≥ 2 GB on 64-bit computers
end

sequences(1:10) % D-by-S array ==> D is number of features 
                %              ==> S is the number of frames of the video   

%% Step 4: Prepare Training Data 

% 0.9 data for Train and 0.1 data for Test
numObservations = numel(sequences);
idx = randperm(numObservations);
N = floor(0.9 * numObservations);

idxTrain = idx(1:N);
sequencesTrain = sequences(idxTrain);
labelsTrain = labels(idxTrain);

idxTest = idx(N+1:end);
sequencesTest = sequences(idxTest);
labelsTest = labels(idxTest);

% Remove Long Sequences

numObservationsTrain = numel(sequencesTrain);
sequenceLengths = zeros(1,numObservationsTrain);

for i = 1:numObservationsTrain
    sequence = sequencesTrain{i};
    sequenceLengths(i) = size(sequence,2);
end

figure
 histogram(sequenceLengths)
  title("Sequence Lengths","FontSize",20)
  xlabel("Sequence Length","FontSize",16)
  ylabel("Frequency","FontSize",16)


maxLength = 300;
idx = sequenceLengths > maxLength;
sequencesTrain(idx) = []; % Remove data longer than 300
labelsTrain(idx) = [];    % Remove labels longer than 300

%% Step 5: Create LSTM Network

numFeatures = size(sequencesTrain{1},1);
numClasses = numel(categories(labelsTrain));

layers = [
    sequenceInputLayer(numFeatures,'Name','sequence')
    bilstmLayer(2000,'OutputMode','last','Name','bilstm')
    dropoutLayer(0.5,'Name','drop')
    fullyConnectedLayer(numClasses,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classification')];

% Specify Training Options
miniBatchSize = 16;
numObservations = numel(sequencesTrain);

options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',1e-4, ...
    'MaxEpochs',30,...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

%% Step 6: Train LSTM Network and Calculate accuracy

if dolstm==1
    [netLSTM,info] = trainNetwork(sequencesTrain,labelsTrain,layers,options);
    netLSTMfile = fullfile(cd,"netLSTM");
    save(netLSTMfile,"netLSTM","info");

elseif dolstm==0
    load("netLSTM.mat")

end

% Calculate the classification accuracy 
YPred = classify(netLSTM,sequencesTest,'MiniBatchSize',miniBatchSize);
YTest = labelsTest;
accuracy = mean(YPred == YTest);

%% Step 7: Assemble Video Classification Network

cnnLayers = layerGraph(netCNN);
layerNames = ["data" "pool5-drop_7x7_s1" "loss3-classifier" "prob" "output"];
cnnLayers = removeLayers(cnnLayers,layerNames);

% Add Sequence Input Layer
inputSize = netCNN.Layers(1).InputSize(1:2);
averageImage = netCNN.Layers(1).Mean;

inputLayer = sequenceInputLayer([inputSize 3], ...
    'Normalization','zerocenter', ...
    'Mean',averageImage, ...
    'Name','input');
layers = [
    inputLayer
    sequenceFoldingLayer('Name','fold')];

lgraph = addLayers(cnnLayers,layers);
lgraph = connectLayers(lgraph,"fold/out","conv1-7x7_s2");

% Add LSTM Layers
lstmLayers = netLSTM.Layers;
lstmLayers(1) = [];
layers = [
    sequenceUnfoldingLayer('Name','unfold')
    flattenLayer('Name','flatten')
    lstmLayers];

lgraph = addLayers(lgraph,layers);
lgraph = connectLayers(lgraph,"pool5-7x7_s1","unfold/in");

lgraph = connectLayers(lgraph,"fold/miniBatchSize","unfold/miniBatchSize");
% Assemble Network
analyzeNetwork(lgraph)
net = assembleNetwork(lgraph)

%% Step 8: Classify Using New Data

filename = "walk-01.avi";
video = readVideo(filename);
numFrames = size(video,4);
figure
for i = 1:numFrames
    frame = video(:,:,:,i);
    imshow(frame/255);
    drawnow
end

video = centerCrop(video,inputSize);
YPred = classify(net,{video})




