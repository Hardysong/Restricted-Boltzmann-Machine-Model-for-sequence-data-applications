function [opts,valid_fields] = dbncreateopts(class,numepochs,batchsize)
%   create a valid opts for the training of DBN
%   The opts struct
%       class : type of dbn, include Gauss-Bernoulli RBM(GBRBM),
%       Bernoulli-Bernoulli RBM(BBRBM) and Continuous RBM(CRBM) of Chen
%       et al., (2003,2002)
%       numepochs : number of epochs
%       batchsize : minibatch size
%       learning_rate_W : learning rate for weight w
%       learning_rate_a : learning rate for parameter a
%       momentum : momentum
%       sig : standard deviation for gaussan noise
%       weightcost : weight cost coefficient
%       reference DeepLearnToolbox

opts = struct;
narginchk(1,3);

if strcmpi(class,'bbrbm')
    opts.class = 'BBRBM';
elseif strcmpi(class,'gbrbm')
    opts.class = 'GBRBM';
elseif strcmpi(class,'crbm')
    opts.class = 'CRBM';
else
    error([mfilename ': Error of the class of RBM, must be <BBRBM/GBRBM/CRBM>']);
end

switch nargin
    case 3
        opts.numepochs = numepochs;
        opts.batchsize = batchsize;
    case 2
        opts.numepochs = numepochs;
        opts.batchsize = 1;
    otherwise
        opts.numepochs = 1;
        opts.batchsize = 1;
end
    
if strcmpi(opts.class,'crbm')
    % parameters define for crbm, according to Chen et al., (2002,2003)
    opts.crbm.learning_rate_W = 0.5;
    opts.crbm.learning_rate_A = 0.5;
    % low and high boundary of sigmoid function
    opts.crbm.thetaL = 0;
    opts.crbm.thetaH = 1;
    % parameter a_j that controls the steepness of the sigmoid function
    opts.crbm.Avis = 1;
    opts.crbm.Ahid = 1;
else
    opts.learning_rate = 0.5;
end

opts.init_type = 'gauss';% gauss or uniform ->initialization method for weight

opts.momentum = 0.9;
opts.sig = 0.2;% used by GBRBM and CRBM
opts.wPenalty = 0.0001;

% Sparseness factor
opts.sparsity = 0;
opts.dropout = 0;
opts.topoMask = [];
opts.sparseGain = 1; % Learning rate gain for sparsity (sigmoid only)

opts.beginAnneal = Inf;
opts.beginWeightDecay = 1; % of epoch to start weight penality

% cost function for monitor the performance of RBM include mse,expll,xent,
% mcxent,correlation
opts.costFun = 'mse'; % default is mean square error
opts.plot = 1;% figure the error plot
opts.CDk = 1; % number of sampled times

opts.verbose = 1; % display progress every verbose

opts.y_train = [];
% reference Y. Bengio et al.,(2007)
opts.partially_supervised = 0; %1/0 
% general set the first layer (for regression problen, crbm) or the last
% layer (for class problem, gb/bb rbm), in this tool, user can set this
% parameter freely.
opts.partially_supervisedlayer = 1;% int scalar or vector define the layer, define from the hidden layer
opts.partially_supervised_type = 'classification'; % classification or regression
opts.partially_supervised_outputFun = 'linear'; % 'linear','exp','sigmoid','softmax','tanh','softrect'
opts.partially_supervised_costFun = 'mse';% available cost function include: mse,expll,xent,mcxent,class,correlation

opts.x_val = [];
opts.y_val = [];

opts.early_stopping = 0;
opts.patience = 5;
opts.test_interval = 1;

% save setting
opts.saveEvery = 100; % save progress every of epochs
opts.saveDir = []; % Path to save the model

valid_fields = fieldnames(opts);
end