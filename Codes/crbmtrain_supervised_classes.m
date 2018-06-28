function crbm = crbmtrain_supervised_classes( crbm,x,opts )
%   Train the RBM combine the target (classification) information.
%   Reference A Fast Learning Algorithm for Deep Belief Nets from Hintion
%   2007
%   Notation:
%           rbm:struct
%           x: input data [num_samples,num_vis]
%           opts:struct
%   author: yibo Sun
%   Date: 2018/06/24

assert(isfloat(x), 'x must be a float');
%assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
m = size(x, 1);

%x = [x repmat([1],m,1)];

[nvis,nhid] = size(crbm.W);

% Loss
loss.type = 'mse';
loss.train.e               = [];
loss.val.e                 = [];

if ~isempty(opts.x_val)
    opts.validation = 1;
else
    opts.validation = 0;
end

earlystop.best_err = Inf;
earlystop.patience  = opts.patience;
earlystop.best_str = '';

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

numbatches = floor(m/opts.batchsize);

ksteps = opts.CDk;

%%%% 2018/06/12 from reference by yibo
batchesCost = zeros(opts.numepochs,1);
crbm.trainCost = nan(opts.numepochs,1);
crbm.stat = zeros(opts.numepochs,1);
crbm.stat2 = zeros(opts.numepochs,1);

% get the target 
y = opts.y_train;

% start the learning process
for i = 1 : opts.numepochs
    kk = randperm(m);
    
    tic;
    % Iterate through data sample
    for l = 1 : numbatches
        
        batch_x = extractminibatch(kk,l,opts.batchsize,x);
        batch_y = extractminibatch(kk,l,opts.batchsize,y);
        
        v0 = batch_x;
        if crbm.dropout > 0
            crbm.dropOutMask = (rand(size(v0,1),nhid) > crbm.dropout);
        end
        h0 = activV2H_supervised(crbm,v0,batch_y);
        
        % Use hidden state to calc the positive direction w and a
        wPos =v0' * h0;
        uPos = h0' * batch_y;
        
        v0Pos = sum(v0,1);
        %         aVPos = v0 .* v0;
        aHPos = sum(h0 .* h0,1);
        h0Pos = sum(h0,1);
        
        % negative phase
        % Note: calculate hidden need visible and target layer,
        [v_state_k,~,t_sample_k] = activH2V_supervised(crbm,h0);
        h_state_k = activV2H_supervised(crbm,v_state_k,t_sample_k);
        % Gibbs sample
        for k = 1 : ksteps - 1
            [v_state_k,~,t_sample_k] = activH2V_supervised(crbm,h_state_k);
            h_state_k = activV2H_supervised(crbm,v_state_k,t_sample_k);
        end
        
        tmp = v_state_k' * h_state_k;
        wNeg = tmp;
        uNeg = h_state_k' * t_sample_k;
        v0Neg = sum(v_state_k,1);
        %
        aHNeg = sum(h_state_k .* h_state_k,1);
        h0Neg = sum(h_state_k,1);
        
        % update weight matrix and activation
        crbm.vW = crbm.vW * crbm.momentum + ...
            crbm.lr_w * ((wPos - wNeg) / size(batch_x,1) - crbm.weightcost*crbm.W);
        crbm.W = crbm.W + crbm.vW;
        % update the bias
        crbm.vb = crbm.momentum * crbm.vb + crbm.lr_w * (v0Pos-v0Neg)' / size(batch_x,1);
        crbm.b = crbm.b + crbm.vb;
        crbm.vc = crbm.momentum * crbm.vc + crbm.lr_w * (h0Pos-h0Neg)' / size(batch_x,1);
        crbm.c = crbm.c + crbm.vc;
        
        % classifier weight update
        crbm.vU = crbm.vU * crbm.momentum + ...
            crbm.lr_w * ((uPos - uNeg) / size(batch_x,1) - crbm.weightcost * crbm.U);
        crbm.U = crbm.U + crbm.vU;
        % classifier bias
        crbm.vd = crbm.momentum * crbm.vd + crbm.lr_w * sum(batch_y-t_sample_k)' / size(batch_x,1);
        crbm.d = crbm.d + crbm.vd;
        
        % calc the mean <.> over the training data,Fun(13) in Chen et al.,2003
        % a_j for hidden unit
        aMean = (aHPos - aHNeg)./size(batch_x,1);
        dA_hid = crbm.lr_a * aMean ./ (size(batch_x,1) .* crbm.Ahid.^2);
        crbm.Ahid = crbm.Ahid + dA_hid;
        %
        delta = v0 - v_state_k;
        batchesCost(i) = batchesCost(i) + sum(sum(delta.^2));
    end
    
    %%%%%%%%%%%%%%%%%%%
    batchesCost(i) = 0.5 * batchesCost(i)/m;
    
    crbm.stat(i) = crbm.W(1,1);
    crbm.stat2(i) = crbm.Ahid(1);
    crbm.trainCost(i) = batchesCost(i);
    
    if opts.validation == 1
        [loss,crbm,~] = crbmmonitor(crbm,loss,x,opts,i);
        str_perf = sprintf(['; Full-batch train ' opts.costFun '= %f, validation ' ...
            opts.costFun ' = %f'], loss.train.e(end),loss.val.e(end));
    else
        [loss,crbm,~] = crbmmonitor(crbm,loss,x,opts,i);
        str_perf = sprintf('; Full-batch train error = %f', loss.train.e(end));
    end

    %loss.train.e(end+1)= err;
    if ishandle(fhandle)
        rbmupdatefigures(fhandle, loss, opts, i);
    end
    
    t = toc;
    disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ', Tool ' num2str(t) 'seconds. ' str_perf]);
    
    if opts.validation == 1
        
        earlystop  = crbmearlystopping(crbm,opts,earlystop,loss,i);
        % stop training?
        if opts.early_stopping && earlystop.patience < 0
            disp('No more Patience. Return best CRBM')
            earlystop.best_val_error = loss.val.e(end);
            earlystop.best_train_error = loss.train.e(end);
            crbm = earlystop.best_crbm;
            %save_model(crbm,opts);
            break;
        end
    end
    
    if crbm.sig > 0.05
        crbm.sig = crbm.sig*1;
    end
    
end
     crbm.loss = loss;
     %save_model(crbm,opts);
end

function save_model(model,opts)
if isempty(opts.saveDir)
    return;
end
disp([mfilename 'Save current Model!']);
if ~isdir(opts.saveDir)
    mkdir(opts.saveDir);
end
fileName = fullfile(opts.saveDir,' model_crbm.mat');
model = model;
save(fileName,'model');
end