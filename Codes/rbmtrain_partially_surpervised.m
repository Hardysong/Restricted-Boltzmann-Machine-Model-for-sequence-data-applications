function rbm = rbmtrain_partially_surpervised( rbm,x,opts )
%   Partially surpervise Train the RBM include Gaussian-Bernoulli RBM and Bernoulli-Bernoulli
%   RBN for Regression or classification problem
%   Notation:
%           crbm:struct
%           x: input data [num_samples,num_vis]
%           opts:struct
%   author: yibo Sun
%   Date: 2017/09/15
%   Modified: 2018/06/12

assert(isfloat(x), 'x must be a float');
assert(all(x(:)>=0) && all(x(:)<=1),'all data in x must be in [0,1]');

m = size(x, 1);

[nvis,nhid] = size(rbm.W);

% Loss
loss.type = opts.costFun;
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
rbm.trainCost = nan(opts.numepochs,1);

% get the target 
y = opts.y_train;

switch lower(rbm.partially_supervised_type)
    case 'regression'
        
        % start the learning process
        for i = 1 : opts.numepochs
            kk = randperm(m);
            tic;
            % Iterate through data sample
            for l = 1 : numbatches
                batch_x = extractminibatch(kk,l,opts.batchsize,x);
                batch_y = extractminibatch(kk,l,opts.batchsize,y);
                
                v0 = batch_x;
                if rbm.dropout > 0
                    rbm.dropOutMask = (rand(size(v0,1),nhid) > rbm.dropout);
                end
                
                [h0,h0_sample] = rbmV2H(rbm,v0,opts);
                
                % compute predictive target from hidden units output (before the sample)
                % Train Partially Supervised according to Bengio
                rbm = rbm_partiallysupervisedtrain(rbm,v0,batch_y,opts);
                
                % Use hidden state to calc the positive direction w and a
                wPos = v0' * h0;
                v0Pos = sum(v0,1);
                h0Pos = sum(h0,1);
                
                % negative phase
                v_k = v0;
                h_state_k = h0_sample;
                for k = 1 : ksteps
                    [v_k,~,~,~] = rbmH2V(rbm,h_state_k,opts);
                    [h_k,h_state_k] = rbmV2H(rbm,v_k,opts);
                end
                
                wNeg = v_k' * h_k;
                v0Neg = sum(v_k,1);
                h0Neg = sum(h_k,1);
                
                % update weight matrix and activation
                dw = wPos - wNeg;
                db = v0Pos-v0Neg;
                if strcmpi(opts.class,'gbrbm')
                    dw = bsxfun(@rdivide,dw,rbm.sig');
                    db = bsxfun(@rdivide,db,rbm.sig.^2);
                end
                rbm.vW = rbm.vW * rbm.momentum + ...
                    rbm.lr_w * (dw / size(batch_x,1) - rbm.weightcost*rbm.W);
                rbm.W = rbm.W + rbm.vW;
                % update the bias
                rbm.vb = rbm.momentum * rbm.vb + rbm.lr_w * db' / size(batch_x,1);
                rbm.b = rbm.b + rbm.vb;
                rbm.vc = rbm.momentum * rbm.vc + rbm.lr_w * (h0Pos-h0Neg)' / size(batch_x,1);
                rbm.c = rbm.c + rbm.vc;
                
                %
                delta = v0 - v_k;
                batchesCost(i) = batchesCost(i) + sum(sum(delta.^2));
                
            end
            
            %%%%%%%%%%%%%%%%%%%
            batchesCost(i) = 0.5 * batchesCost(i)/m;
            
            rbm.trainCost(i) = batchesCost(i);
            
            if opts.validation == 1
                [loss,rbm,~] = rbmmonitor(rbm,loss,x,opts,i);
                str_perf = sprintf(['; Full-batch train ' opts.costFun '= %f, validation ' ...
                    opts.costFun ' = %f'], loss.train.e(end),loss.val.e(end));
            else
                [loss,rbm,~] = rbmmonitor(rbm,loss,x,opts,i);
                str_perf = sprintf('; Full-batch train error = %f', loss.train.e(end));
            end
            
            %loss.train.e(end+1)= err;
            if ishandle(fhandle)
                rbmupdatefigures(fhandle, loss, opts, i);
            end
            
            t = toc;
            if ~mod(i,opts.verbose)
                disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ', Tool ' num2str(t) 'seconds. ' str_perf]);
            end
            
            if opts.validation == 1
                
                earlystop  = rbmearlystopping(rbm,opts,earlystop,loss,i);
                % stop training?
                if opts.early_stopping && earlystop.patience < 0
                    disp('No more Patience. Return best CRBM')
                    earlystop.best_val_error = loss.val.e(end);
                    earlystop.best_train_error = loss.train.e(end);
                    rbm = earlystop.best_rbm;
                    %save_model(rbm,opts);
                    break;
                end
            end
            
            if isfield(rbm,'sig') && rbm.sig > 0.05
                rbm.sig = rbm.sig*1;
            end
            
        end
        rbm.loss = loss;
        %save_model(rbm,opts);
    case 'classification'
        
        % start the learning process
        for i = 1 : opts.numepochs
            kk = randperm(m);
            tic;
            % Iterate through data sample
            for l = 1 : numbatches
                batch_x = extractminibatch(kk,l,opts.batchsize,x);
                batch_y = extractminibatch(kk,l,opts.batchsize,y);
                
                v0 = batch_x;
                if rbm.dropout > 0
                    rbm.dropOutMask = (rand(size(v0,1),nhid) > rbm.dropout);
                end
                [h0,h0_sample] = rbmV2H(rbm,v0,opts,batch_y);
                
                % Use hidden state to calc the positive direction w and a
                wPos = v0' * h0;
                uPos = h0' * batch_y;
                v0Pos = sum(v0,1);
                h0Pos = sum(h0,1);
                
                % negative phase
                v_state_k = v0;
                h_state_k = h0_sample;
                for k = 1 : ksteps
                    [v_k,v_state_k,t_k,t_sample_k] = rbmH2V(rbm,h_state_k,opts);
                    [h_k,h_state_k] = rbmV2H(rbm,v_k,opts,t_k);
                end
                
                wNeg = v_k' * h_k;
                uNeg = h_k' * t_sample_k;
                v0Neg = sum(v_k,1);
                h0Neg = sum(h_k,1);
                
                % update weight matrix and activation
                dw = wPos - wNeg;
                db = v0Pos-v0Neg;
                if strcmpi(opts.class,'gbrbm')
                    dw = bsxfun(@rdivide,dw,rbm.sig);
                    db = bsxfun(@rdivide,db,rbm.sig.^2);
                end
                rbm.vW = rbm.vW * rbm.momentum + ...
                    rbm.lr_w * (dw / size(batch_x,1) - rbm.weightcost*rbm.W);
                rbm.W = rbm.W + rbm.vW;
                % update the bias
                rbm.vb = rbm.momentum * rbm.vb + rbm.lr_w * db' / size(batch_x,1);
                rbm.b = rbm.b + rbm.vb;
                rbm.vc = rbm.momentum * rbm.vc + rbm.lr_w * (h0Pos-h0Neg)' / size(batch_x,1);
                rbm.c = rbm.c + rbm.vc;
                
                % classifier weight update
                rbm.vU = rbm.vU * rbm.momentum + ...
                    rbm.lr_w * ((uPos - uNeg) / size(batch_x,1) - rbm.weightcost * rbm.U);
                rbm.U = rbm.U + rbm.vU;
                % classifier bias
                rbm.vd = rbm.momentum * rbm.vd + rbm.lr_w * sum(batch_y-t_sample_k)' / size(batch_x,1);
                rbm.d = rbm.d + rbm.vd;
                %
                delta = v0 - v_k;
                batchesCost(i) = batchesCost(i) + sum(sum(delta.^2));
                
            end
            
            %%%%%%%%%%%%%%%%%%%
            batchesCost(i) = 0.5 * batchesCost(i)/m;
            
            rbm.trainCost(i) = batchesCost(i);
            
            if opts.validation == 1
                [loss,rbm,~] = rbmmonitor(rbm,loss,x,opts,i);
                str_perf = sprintf(['; Full-batch train ' opts.costFun '= %f, validation ' ...
                    opts.costFun ' = %f'], loss.train.e(end),loss.val.e(end));
            else
                [loss,rbm,~] = rbmmonitor(rbm,loss,x,opts,i);
                str_perf = sprintf('; Full-batch train error = %f', loss.train.e(end));
            end
            
            %loss.train.e(end+1)= err;
            if ishandle(fhandle)
                rbmupdatefigures(fhandle, loss, opts, i);
            end
            
            t = toc;
            if ~mod(i,opts.verbose)
                disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) ', Tool ' num2str(t) 'seconds. ' str_perf]);
            end
            
            if opts.validation == 1
                
                earlystop  = rbmearlystopping(rbm,opts,earlystop,loss,i);
                % stop training?
                if opts.early_stopping && earlystop.patience < 0
                    disp('No more Patience. Return best CRBM')
                    earlystop.best_val_error = loss.val.e(end);
                    earlystop.best_train_error = loss.train.e(end);
                    rbm = earlystop.best_rbm;
                    %save_model(rbm,opts);
                    break;
                end
            end
            
            if isfield(rbm,'sig') && rbm.sig > 0.05
                rbm.sig = rbm.sig*1;
            end
            
        end
        rbm.loss = loss;
        %save_model(rbm,opts);
end

end

function save_model(model,opts)
if isempty(opts.saveDir)
    return;
end
disp([mfilename ', Save current Model!']);
if ~isdir(opts.saveDir)
    mkdir(opts.saveDir);
end
fileName = fullfile(opts.saveDir,' model_crbm.mat');
model = model;
save(fileName,'model');
end