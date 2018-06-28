function [rbm,bp_info] = crbm_partiallysupervisedtrain(rbm,x,y,opts)
% --------------------------------------------------------------------------
%   Used by Continuous RBM
%   This function is called by the procedure of unsuperivesd DBN training
%   use supervised (target information) stochastic gradient descant updates
%   with respect to cost Function. Creating temporary output weights U to
%   map the hidden layer outputs to predicted targets. 
%   This manner deal with uncooperative input distributions, that the
%   supervised training process could force the weight between vis anbd hid
%   represent the target better.
%   Reference to 'Greedy Layer-Wise Training of Deep Networks' Bengio2007
%  -----------------------------------------------------------------------------
%   Author: Yibo Snn

narginchk(4,4);

% Run a Back-Propagation process
[rbm,bp_info] = rbm_nn_fp(rbm,x,y);

bp_info = rbm_nn_bp(rbm,bp_info);
rbm = rbm_nn_updateParams(rbm,bp_info);


end

function [rbm,bp_info] = rbm_nn_fp( rbm,netInput,targets )
%   --------------------------------------------------------------------
%   Forward Propagation of input singals, Also updates states of network
%   cost if provided with <targets>
%   input:
%       mlnn: model;
%       opts: setup of the model
%       netInput: input variable
%       targets: target variable
%
%   author: Yibo Sun
bp_info = struct;

nObs = size(netInput,1);

bp_info.act{1} = netInput;

% feedforward pass
% Hidden Layer Pre-Activation
preAct = bsxfun(@plus,bp_info.act{1}*rbm.W,rbm.c');

A = repmat(rbm.Ahid,nObs,1);
% Hidden Layer outputs
V = sigmFun(rbm.thetaL,rbm.thetaH,A,preAct);

% Dropout
if rbm.dropout > 0
    V = V .* rbm.dropOutMask;
end

bp_info.act{2} = V;

% Output Layer Pre-Activation
preAct = bsxfun(@plus,V*rbm.U,rbm.d');

supervised_output = calcAct(preAct,rbm.outputFun);
bp_info.act{3} = supervised_output;
% training cost
[J,dJ] = cost(supervised_output,targets,rbm.supervised_costFun);
bp_info.netError = dJ;
bp_info.targetError = J;

end

function out = calcAct(in,actFun)
% output from the active function
%-------------------------------------------------------------
% Available activation function include
% 'linear','exp','sigmoid','softmax','tanh','softrect'
%-------------------------------------------------------------
switch lower(actFun)
    case 'linear'
        out = stabilizeInput(in);
    case 'exp'
        in = stabilizeInput(in);
        out = exp(in);
    case 'sigmoid'
        in = stabilizeInput(in);
        out = 1 ./ (1+exp(-in));
    case 'softmax'
        in = stabilizeInput(in);
        maxIn = max(in,[],2);
        tmp = exp(bsxfun(@minus,in,maxIn));
        out = bsxfun(@rdivide,tmp,sum(tmp,2));
    case 'tanh'
        in = stabilizeInput(in);
        out = tanh(in);
    case 'softrect'
        k=8;
        in = stabilizeInput(in,k);
        out = 1/k.*log(1+exp(k*in));
end
        
end

function in = stabilizeInput(in,k)
% To make the numerical stability
%---------------------------------------------------------------
% Utility function to ensure numerical stability, Clip values of <in> such
% that exp(k*in) is within single numerical precision
%--------------------------------------------------------------
if nargin < 2
    k = 1;
end
cutoff = log(realmin('single'));
in(in*k>-cutoff) = -cutoff/k;
in(in*k<cutoff) = cutoff/k;

end

function [J,dJ] = cost(netOut,targets,costFun)
% calculate the cost and gradient
%----------------------------------------------------------------
% Calculate the output error <J> and error gradients <dJ> based on
% available cost function (mse,expll,xent,mcxent,class,correlation), Note
% that 'correlation' and 'class' do not provide gradient
nObs = size(netOut,1);

switch lower(costFun)
    case 'mse'
        % mean squared error (linear regression)
        delta = targets - netOut;
        J = 0.5*sum(sum(delta.^2))./nObs;
        dJ = -delta;
    case 'expll'
        % exponential log likelihood (poisson regression)
        J = sum(sum((netOut - targets.*log(netOut))))/nObs;
        dJ = 1 - targets./netOut;
    case 'xent'
        % ½»²æìØÎó²î cross entropy (binary classification/Logistic regression)
        J = -sum(sum(targets.*log(netOut) + (1-targets).*log(1-netOut+0.0001)))/nObs;
        dJ = (netOut - targets)./(netOut.*(1-netOut)+0.0001);
    case 'mcxent'
        % multi-class cross entropy (classification) - softmax
        J = -sum(sum(targets.*log(netOut)))./nObs;
        dJ = netOut - targets;  % modified the original error from 'medal tool'
    case {'class','classerr'}
        % classification error (winner take all)
        [~,c] = max(netOut,[],2);
        [~,t] = max(targets,[],2);
        J = sum(sum((c~=t)))/nObs;
        dJ = 'no gradient';
    case {'cc','correlation'}
        J = corr2(netOut(:),targets(:));
        dJ = 'no gradient';
end
end

%% 
function bp_info = rbm_nn_bp( rbm,bp_info )
%--------------------------------------------------
%   Perform gradient descent on the loss w.r.t each of the model parameters
%   using the backpropagation algorithm. Return bp information for update
%   the RBM
%   Author: Yibo Sun
%--------------------------------------------------

% derivative of output activation function
dAct = calcActDerive(bp_info.act{3},rbm.outputFun);

% Error partial derivatives
dE{3} = bp_info.netError.*dAct;

% Backpropagate error algorithm
% Layer error contribution % Error back-propagation
propError = dE{3}*rbm.U';

% Derivative of Activation Function
asy = rbm.thetaH - rbm.thetaL;
dAct = calcActDerive(bp_info.act{2},'asy_sigmoid',asy);

% Calculate layer error signal (include sparsity)
dE{2} = propError.*dAct;
if rbm.dropout > 0 % add by yibo Sun.
    dE{2} = dE{2} .* rbm.dropOutMask;
end

% Calculate dE/dW for each layer
for lL = 1 : 2
    bp_info.dW{lL} = (dE{lL+1}'*bp_info.act{lL})'/size(dE{lL+1},1);
    bp_info.db{lL} = sum(dE{lL+1})'/size(dE{lL+1},1);
    if isfield(rbm,'normGrad') && rbm.normGrad
        % Constrain gradients to have unit norm 
        normdW = norm([bp_info.dW{lL}(:);mlnn.db{lL}(:)]);
        bp_info.dW{lL} = bp_info.dW{lL}/normdW;
        bp_info.db{lL} = bp_info.db{lL}/normdW;
    end
end

end

function dAct = calcActDerive(in,actFun,asy)
%------------------------------------------------------
%   Calculate the output activation derivatives <dAct> from an input <in>
%   for activation function <actFun>. Available activation function
%   derivatives include 'linear',¡®softmax¡¯,'exp','sigmoid','tanh',and 'softrect'.
%-------------------------------------------------------

switch lower(actFun)
    case {'linear','softmax'}
        dAct = ones(size(in));
    case 'exp'
        in = stabilizeInput(in,1);
        dAct = in;
    case 'sigmoid'
        in = stabilizeInput(in,1);
        dAct = in .* (1 - in);
    case 'asy_sigmoid'
        in = stabilizeInput(in,1);
        dAct = asy * in .* (1 - in);
    case 'tanh'
        in = stabilizeInput(in,1);
        dAct = 1 - in.^2;
    case 'softrect'
        k=8;
        in = stabilizeInput(in,k);
        dAct = 1./(1+exp(-k*in));
    otherwise
        error([mfilename 'Output error!']);
end

end

function rbm = rbm_nn_updateParams(rbm,bp_info)
%-----------------------------------------------
%   Update netword parameters based on the gradient, perform regularization
%   such as weight decay and weight rescale
%-----------------------------------------------
%   Author: Yibo Sun
momentum = rbm.momentum;

if rbm.weightcost > 0
    % L2-weight decay
    wPenalty{1} = rbm.W * rbm.weightcost;
    wPenalty{2} = rbm.U * rbm.weightcost;
elseif rbm.weightcost < 0
    % (Approximate) L1-weight decay
    wPenalty{1} = sign(rbm.W) * rbm.weightcost;
    wPenalty{2} = sign(rbm.U) * rbm.weightcost;
end

% Update weight and bias (momentum)
rbm.vW = momentum * rbm.vW + rbm.lr_w * (bp_info.dW{1} + wPenalty{1});
rbm.vU = momentum * rbm.vU + rbm.lr_w * (bp_info.dW{2} + wPenalty{2});
rbm.vc = momentum * rbm.vc + rbm.lr_w * bp_info.db{1};
rbm.vd = momentum * rbm.vd + rbm.lr_w * bp_info.db{2};

% Update weights
rbm.W = rbm.W - rbm.vW;
rbm.c = rbm.c - rbm.vc;
rbm.d = rbm.d - rbm.vd;

% Constrain norm of input weights to be within a Ball of radius
% net.maxNorm
if isfield(rbm,'maxNorm') && rbm.maxNorm < Inf
    rescale = sqrt(sum(rbm.W.^2,2));
    mask = rescale > rbm.maxNorm;
    rescale(~mask) = 1;
    rescale(mask) = rescale(mask)/rbm.maxNorm;
    
    rbm.W = bsxfun(@rdivide,rbm.W,rescale);
    
    rescale = sqrt(sum(rbm.U.^2,2));
    mask = rescale > rbm.maxNorm;
    rescale(~mask) = 1;
    rescale(mask) = rescale(mask)/rbm.maxNorm;
    rbm.U = bsxfun(@rdivide,rbm.U,rescale);
end

end