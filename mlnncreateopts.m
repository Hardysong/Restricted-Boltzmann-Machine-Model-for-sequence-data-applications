function [opts,valid_fields] = mlnncreateopts(numepochs,batchsize)
%   create a valid opts for the training of Mulit-Layer neural networks
%   Supports multiple activation functions including linear, sigmoid, tanh,
%   exponential, and soft Rectification (softplus)
%   
%   Support cost function include mean squared error (mse), binary (xent) 
%   and multi-class cross-entroy (mcxent), and exponential log likelihood
%   (expll)
%   
%   Support multiple regularization techniques including weight decay,
%   hidden unit dropout, and early stopping
%
%   The opts struct
%       numepochs : number of epochs
%       batchsize : minibatch size
%       learning_rate : learning rate
%       momentum : momentum
%       costFun : COST Function
%       normGrad : constrain gradients to have unit norm
%       
%       weightcost : weight cost coefficient

opts = struct;

switch nargin
    case 2
        opts.numepochs = numepochs;
        opts.batchsize = batchsize;
    case 1
        opts.numepochs = numepochs;
        opts.batchsize = 1;
    otherwise
        opts.numepochs = 1;
        opts.batchsize = 1;
end
    
opts.learning_rate = 0.1;

opts.momentum = 0.9;

opts.wPenalty = 0.00001; % < 0 means L1 weight penalty

opts.actFun = {'sigmoid'}; 

opts.costFun = 'mse';
opts.normGrad = 0;
opts.maxNorm = Inf;  % Renormalize weight whose nor exceed maxNorm (typical value 3, 4) reference: http://cs231n.github.io/neural-networks-2/#reg

opts.sparsity = 0;  % target sparisty (sigmoid only)
opts.sparseGain = 1; % gain on learning rate for sparsity (sigmoid only)
opts.dropout = 0;  % proportion of hidden unit dropout
opts.denoise = 0; % proportion of visible unit dropout

opts.plot = 1;% figure the error plot
opts.display_interval = 1;

opts.x_val = [];
opts.y_val = [];
opts.early_stopping = 0;
opts.patience = 5;
opts.test_interval = 1; % used by early stop

opts.saveEvery = 10;
opts.saveDir = [];

valid_fields = fieldnames(opts);

end