function [ mlnn,loss ] = mlnn_fp( mlnn,netInput,targets )
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

narginchk(2,3);
loss = [];

if ~exist('targets','var')
    targets = [];
    mlnn.testing=1;
end

nObs = size(netInput,1);
mlnn.act{1} = netInput;

% feedforward pass
for i = 2 : mlnn.n
    % Layer Pre-Activation
    preAct = bsxfun(@plus,mlnn.act{i-1}*mlnn.W{i-1}',mlnn.b{i-1}');
    
    % Layer activation
    mlnn.act{i} = calcAct(preAct,mlnn.actFun{i-1});

    % Dropout
    if mlnn.dropout > 0 && i < mlnn.n
        if mlnn.testing % reweight active during testing using dropout
            mlnn.act{i} = mlnn.act{i} .* (1-mlnn.dropout);
        else
            mlnn.dropOutMask{i} = (rand(size(mlnn.act{i})) > mlnn.dropout);
            mlnn.act{i} = mlnn.act{i} .* mlnn.dropOutMask{i};
        end
    end
    
    if numel(find(mlnn.act{i}==1))>0
        disp('');
    end
    
    % moving average for target sparsity (sigmoid only)
    if strcmpi('sigmoid',mlnn.actFun{i-1})
        if mlnn.sparsity > 0
            mlnn.meanAct{i} = 0.9 * mlnn.meanAct{i} + 0.1 * mean(mlnn.act{i});
        end
    end
end
mlnn.testing=0;

% training cost
if ~isempty(targets)
    [J,dJ] = cost(mlnn,targets,mlnn.costFun);
    mlnn.J = J;
    mlnn.netError = dJ;
    loss = J;
end

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

function [J,dJ] = cost(mlnn,targets,costFun)
% calculate the cost and gradient
%----------------------------------------------------------------
% Calculate the output error <J> and error gradients <dJ> based on
% available cost function (mse,expll,xent,mcxent,class,correlation), Note
% that 'correlation' and 'class' do not provide gradient
netOut = mlnn.act{end};

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