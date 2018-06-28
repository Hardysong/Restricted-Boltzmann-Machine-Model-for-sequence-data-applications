function [ mlnn, Loss ] = mlnntrain( mlnn, train_x, train_y, opts )
%   TRAIN a Multi-layer neural networks
%   -------------------------------------------------
%   Train a neural networks usint stochastic gradient descent. 
%   Input:
%       mlnn : neural network model, contains weights, bias, active Fun et
%       al.
%       train_x: training input features
%       train_y: labels or targets
%       opts: training setting
%   Output:
%       mlnn : model after training, added train_error (each epoch),
%       validation error (each epoch), 
%       Loss : Loss during training
%   Author: Yibo Sun
assert(isfloat(train_x),'train_x must be a float');
assert(nargin == 4, 'number of input argument must be 4 (model,x,y,opts)');

loss.type = mlnn.costFun;
loss.train.e = [];
loss.train.e_frac = [];
loss.val.e = [];
loss.val.e_frac = [];

if ~isempty(opts.x_val) && ~isempty(opts.y_val)
    mlnn.validation = 1;% validation the model during the training process
else
    mlnn.validation = 0;
end

earlystop.best_err = Inf;
earlystop.patience  = opts.patience;
earlystop.best_str = '';

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

mlnn.trainCost = zeros(opts.numepochs,1);
if mlnn.validation == 1
    opts.validation = 1;
    val_x = opts.x_val;
    val_y = opts.y_val;
else
    opts.validation =0;
end

m = size(train_x,1);
numbatches = floor(m/opts.batchsize);

for i = 1 : opts.numepochs
    tic;
    
    kk = randperm(m);
    % Iterate through data sample
    mlnn.epoch = i;
    batchesCost = zeros(numbatches,1);
    for l = 1 : numbatches
        batch_x = extractminibatch(kk,l,opts.batchsize,train_x);
        batch_y = extractminibatch(kk,l,opts.batchsize,train_y);
        
        % Denoise AutoEncode (AdE) neural networks used ,Add binary noise
        % to input
        if opts.denoise > 0
            batch_x = batch_x .* (rand(size(batch_x))>opts.denoise);
        end
        
        % Back Propagation algorithm
        mlnn = mlnn_fp(mlnn,batch_x,batch_y);
        mlnn = mlnn_bp(mlnn);
        mlnn = mlnn_updateParams(mlnn);
        
        batchesCost(l) = mlnn.J;
    end
    
    mlnn.trainCost(i) = mean(batchesCost);
    
    if mlnn.validation == 1
        loss = nnmonitor(mlnn,loss,train_x,train_y,val_x,val_y);
        str_perf = sprintf(['; Full-batch train ' mlnn.costFun '= %f, validation ' ...
            mlnn.costFun ' = %f'], loss.train.e(end),loss.val.e(end));
    else
        loss = nnmonitor(mlnn,loss,train_x,train_y);
        str_perf = sprintf('; Full-batch train error = %f', loss.train.e(end));
    end
    
    if ishandle(fhandle) && ~mod(i,opts.display_interval) && i > 1
        monitorupdatefigures(fhandle, loss, opts, i);
    end
    
    t = toc;
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ', Tool ' num2str(t) 'seconds. ' str_perf]);
    
    % Save current model
    if ~mod(i,opts.saveEvery) && ~isempty(opts.saveDir)
        save_model(mlnn,opts);
    end
    
    if isfield(opts,'validation') && opts.validation == 1
        
        earlystop  = earlystopping(mlnn,opts,earlystop,loss,i);
        % stop training?
        if opts.early_stopping && earlystop.patience < 0
            disp('No more Patience. Return the best Model')
            earlystop.best_val_error = loss.val.e(end);
            earlystop.best_train_error = loss.train.e(end);
            mlnn = earlystop.best_model;
            save_model(mlnn,opts);
            break;
        end
    end
    
end
Loss = loss;
end

function save_model(model,opts)
if isempty(opts.saveDir)
    return;
end
disp([mfilename 'Save current Model!']);
if ~isdir(opts.saveDir)
    mkdir(opts.saveDir);
end
fileName = fullfile(opts.saveDir,'model_saved.mat');
model = model;
save(fileName,'model');
end