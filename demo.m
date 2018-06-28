%% demo 1£ºcontinuous rbm 
rand('state',0);

Ndata = 500;
[X_dat,Y_dat] = GenData(Ndata);
dat = [X_dat Y_dat];
y_dat = GenData(Ndata);
%
% dat = rand(Ndata,12);
% y_dat = rand(Ndata,1);

[opts,valid_fields] = dbncreateopts('crbm',500,100);
opts.CDk = 50; % number of Gibbs sample
opts.plot = 1;
opts.crbm.Avis = 0.1; % define for this test
opts.batchsize= 50;
opts.numepochs = 1000;
opts.saveDir = 'E:\research_work_5\ContinuousRBM_v2\models';

opts.y_train = y_dat;
opts.partially_supervised = 1;% 1 -> partlly supervised or not
opts.partially_supervisedlayer = [1];
opts.partially_supervised_type = 'regression'; % classification or regression

dbn = dbnsetup([8,8],dat,opts);
dbn = dbntrain(dbn,dat,opts);

figure;subplot(1,2,1);plot(1:length(dbn.crbm{1}.loss.train.e),...
    dbn.crbm{1}.loss.train.e,'r-');
subplot(1,2,2);plot(1:length(dbn.crbm{1}.loss.val.e),...
    dbn.crbm{1}.loss.val.e,'r-');
%subplot(1,2,1);plot(cdbn.crbm{1}.trainCost);
figure;plot(X_dat,Y_dat,'bo');title(['ksteps = ' num2str(opts.CDk)]);
x_reconstruct = reconstruct(dbn.crbm{1},dat,opts);
X_rec = x_reconstruct(:,1);
Y_rec = x_reconstruct(:,2);
hold on;plot(X_rec,Y_rec,'r.');

% test new data
nSamples = 500;
input = rand(nSamples,2);
datout = reconstruct(dbn.crbm{1},input,opts);
%figure;
plot(datout(:,1),datout(:,2),'g*');

dscatter(datout(:,1),datout(:,2));

figure;subplot(1,2,1);plot(dbn.crbm{1}.stat);subplot(1,2,2);plot(dbn.crbm{1}.stat2);


%% demo2: test the cdbn for classification problem
load mnistSmall;

train_x = double(trainData);
test_x  = double(testData);
train_y = double(trainLabels);
test_y  = double(testLabels);

[opts,valid_fields] = dbncreateopts('crbm',10,100);
opts.numepochs = 100;
opts.crbm.Avis = 1;
opts.plot = 1;
opts.batchsize= 100;
opts.numepochs = 100;
opts.saveDir = 'E:\research_work_5\ContinuousRBM\models';

opts.y_train = train_y;
opts.partially_supervised = 1;
opts.partially_supervisedlayer = [2];
opts.partially_supervised_type = 'classification'; % classification or regression
opts.partially_supervised_outputFun = 'softmax';

sizes = [100,100];
rand('state',0);
dbn = dbnsetup(sizes,train_x,opts);
dbn = dbntrain(dbn, train_x, opts);


%% demo 3: test the dbn consist by GBRBM for classification or regression problem
% In my testing, I found that the Gaussian mode is very sensible to the
% previous learned gradient (vW, vd ...), that means big momentum or
% learning rate will lead the learning process unstable or the lower
% boundary of KL divergence become diffuse. So small learning rate, momentum and bigger
% weight penalty is necessary for Gaussian input 
load mnistSmall;

train_x = double(trainData);
test_x  = double(testData);
train_y = double(trainLabels);
test_y  = double(testLabels);

[opts,valid_fields] = dbncreateopts('gbrbm',10,100);
opts.numepochs = 100;
opts.plot = 1;
opts.batchsize= 100;
opts.numepochs = 100;
opts.wPenalty = 0.002;
opts.learning_rate=0.002;
opts.momentum=0;
opts.saveDir = 'E:\research_work_5\ContinuousRBM\models';

% if training the model combine target information need define follow:
opts.y_train = train_y;
opts.partially_supervised = 1;
opts.partially_supervisedlayer = [2];% define which layer to execute partlly supervised training
opts.partially_supervised_type = 'classification'; % classification or regression
opts.partially_supervised_outputFun = 'softmax';

sizes = [100,100];
rand('state',0);
dbn = dbnsetup(sizes,train_x,opts);
dbn = dbntrain(dbn, train_x, opts);

%% demo 4: test the dbn consist by BBRBM for classification or regression problem

load mnistSmall;

train_x = double(trainData);
test_x  = double(testData);
train_y = double(trainLabels);
test_y  = double(testLabels);

[opts,valid_fields] = dbncreateopts('bbrbm',10,100);
opts.numepochs = 100;
opts.plot = 1;
opts.batchsize= 100;
opts.numepochs = 100;
opts.wPenalty = 0.002;
opts.learning_rate=0.002;
opts.momentum=0.8;
opts.saveDir = 'E:\research_work_5\ContinuousRBM_v2\models';

% if training the model combine target information need define follow:
opts.y_train = train_y;
opts.partially_supervised = 1;
opts.partially_supervisedlayer = [2];% define which layer to execute partlly supervised training
opts.partially_supervised_type = 'classification'; % classification or regression
opts.partially_supervised_outputFun = 'softmax';

sizes = [100,100];
rand('state',0);
dbn = dbnsetup(sizes,train_x,opts);
dbn = dbntrain(dbn, train_x, opts);

%% demo 5: test the GBRBM using another dataset only unsupervised method
load('facesDataGray.mat');
data = bsxfun(@minus,data,mean(data));
data = bsxfun(@rdivide,data,std(data));

[opts,valid_fields] = dbncreateopts('gbrbm',10,100);
opts.numepochs = 100;
opts.plot = 1;
opts.batchsize= 100;
opts.numepochs = 100;
opts.wPenalty = 0.002;
opts.learning_rate=0.002;
opts.momentum=0.5;
opts.saveDir = 'E:\research_work_5\ContinuousRBM\models';


sizes = [1000];
rand('state',0);
dbn = dbnsetup(sizes,data,opts);
dbn = dbntrain(dbn, data, opts);

%% demo 6: test the multi-layer neural networks
load mnistSmall.mat;
mlnn_arch = mlnncreateopts(100,100);
mlnn_arch.actFun={'sigmoid','softmax'};
mlnn_arch.normGrad = 0;
mlnn_arch.costFun = 'mse';
mlnn_arch.dropout = 0.5;
mlnn_arch.wPenalty=0;
mlnn = mlnnsetup([28*28,100,10],mlnn_arch);
train_x = trainData;
train_y = single(trainLabels);
% mlnn_arch.x_val = testData;
% mlnn_arch.y_val = single(testLabels);
rand('state',0);
[ mlnn, Loss ] = mlnntrain( mlnn, train_x, train_y, mlnn_arch );

mlnn.costFun = 'class';
x_val = testData;
y_val = single(testLabels);
[~,err] = mlnn_fp( mlnn,x_val,y_val );