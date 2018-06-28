function [ V,V_sample,T_p,T_sample] = rbmH2V( rbm,H,opts,sample )
%----------------------------------------------------------------------
%   Used by GBRBM and BBRBM
%   Activate:
%           h2v: update visible unit states conditional on the current
%           states of the hidden units
%           input:
%               crbm: struct
%               H: Hidden layer sample
%           output:
%               V: visible output
%               V_sample: sample from V
%               T_p: probability of Target output, if available
%               T_sample: sample the Target, if available
%   Author: Yibo Sun
%   2018/06/25

narginchk(3,4);

if nargin == 3
    sample = 1;
end

nObs = size(H,1);

V_sample = [];
T_p=[];
T_sample=[];

switch lower(opts.class)
    case 'bbrbm'
        V = sigmoid(bsxfun(@plus,H*rbm.W',rbm.b'));
        if sample
            V_sample = V > rand(size(V));
        else
            V_sample = [];
        end
    case 'gbrbm'
        h = bsxfun(@times,H*rbm.W',rbm.sig);
        V = bsxfun(@plus,h,rbm.b');
        
        if sample
            % Draw sample from a multivariate normal with mean <V> and
            % identitu convariance
            V_sample = mvnrnd(V,ones(1,size(V,2)));
        else
            V_sample = [];
        end
end

if isfield(rbm,'partially_supervised_type') && strcmpi(rbm.partially_supervised_type,'classification')
    T_p = calcAct(bsxfun(@plus,H * rbm.U, rbm.d'),rbm.outputFun);
    T_sample = sampleClasses(T_p);
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

function classes = sampleClasses(pClass)
%----------------------------------------------------------------
% Sample class labels <classes> given  a set of class probabilities <pClass>
% using uniform distribution
%----------------------------------------------------------------
% Reference medal-master
[nObs,nClass] = size(pClass);
classes = zeros(size(pClass));

% ensure each row of x is normalized.
pClass = bsxfun(@rdivide,pClass,sum(pClass,2));
for iC = 1 : nObs
    probs = pClass(iC,:);
    samp = cumsum(probs) > rand;
    idx = min(find(max(samp)==samp));
    classSamp = zeros(size(probs));
    classSamp(idx) = 1;
    classes(iC,:) = classSamp;
end
end