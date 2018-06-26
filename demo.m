rand('state',0);

Ndata = 500;
[X_dat,Y_dat] = GenData(Ndata);
dat = [X_dat Y_dat];
y_dat = GenData(Ndata);

[opts,valid_fields] = dbncreateopts('crbm',500,100);
opts.CDk = 50; % number of Gibbs sample
opts.plot = 0;
opts.crbm.Avis = 0.1;
opts.batchsize= 50;
opts.numepochs = 1000;
opts.saveDir = 'E:\research_work_5\ContinuousRBM\models';
opts.y_train = y_dat;
opts.partially_supervised = 1;
opts.partially_supervisedlayer = [1];
opts.partially_supervised_type = 'regression'; % classification or regression

cdbn = dbnsetup([8],dat,opts);
cdbn = dbntrain(cdbn,dat,opts);

figure;subplot(1,2,1);plot(1:length(cdbn.crbm{1}.loss.train.e),...
    cdbn.crbm{1}.loss.train.e,'r-');
subplot(1,2,2);plot(1:length(cdbn.crbm{1}.loss.val.e),...
    cdbn.crbm{1}.loss.val.e,'r-');
%subplot(1,2,1);plot(cdbn.crbm{1}.trainCost);
figure;plot(X_dat,Y_dat,'bo');title(['ksteps = ' num2str(opts.CDk)]);
x_reconstruct = reconstruct(cdbn.crbm{1},dat,opts);
X_rec = x_reconstruct(:,1);
Y_rec = x_reconstruct(:,2);
hold on;plot(X_rec,Y_rec,'r.');

% test new data
nSamples = 500;
input = rand(nSamples,2);
datout = reconstruct(cdbn.crbm{1},input,opts);
%figure;
plot(datout(:,1),datout(:,2),'g*');

% [x_in,y_in] = GenData(nSamples);
% dat_rec = reconstruct(cdbn.crbm{1},[x_in,y_in],opts);
% plot(dat_rec(:,1),dat_rec(:,2),'g*');
dscatter(datout(:,1),datout(:,2));

figure;subplot(1,2,1);plot(cdbn.crbm{1}.stat);subplot(1,2,2);plot(cdbn.crbm{1}.stat2);


% test the cdbn
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
cdbn = dbnsetup(sizes,train_x,opts);
cdbn = dbntrain(cdbn, train_x, opts);
