function [V,T_p,T_sample] = activH2V_supervised( crbm, H)
%   Activate:
%           h2v: using hidden to calculate the state of visible 
%           input:
%               crbm: struct
%               H: Hidden layer sample
%           output:
%               V: visible output
%               T_p: probability of Target output
%               T_sample: sample the Target
%   Author: Yibo Sun
%   2018/06/25

narginchk(2,2);

% set the low and high boundary layer of sigmoid function
thetaL = crbm.thetaL;
thetaH = crbm.thetaH;

num = size(H,1);

H1 = H * crbm.W' + repmat(crbm.b',num,1);

V = H1 + crbm.sig * randn(size(H1));

A = repmat(crbm.Avis,num,1);

V = sigmFun(thetaL,thetaH,A,V);

T1 = bsxfun(@plus,H * crbm.U, crbm.d');

T_p = calcAct(T1,crbm.outputFun);

T_sample = sampleClasses(T_p);

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