function mlnn = mlnnsetup( sizes,opts )
%   setup the structe of the multi-layer neural networks
%   mlnnsetup struct:
%       size : [number of input feature, hidden, number of output]
%       opts : struct for setup mlnn
%      
%   author: yibo Sun
mlnn.size = sizes;
mlnn.n = numel(mlnn.size);

if numel(opts.actFun) == 1
    mlnn.actFun = {repmat(opts.actFun,1,mlnn.n-1)};
elseif numel(opts.actFun) == mlnn.n-1
    mlnn.actFun = opts.actFun;
else
    error(['Activation function error!']);
end
mlnn.costFun = opts.costFun;
if numel(opts.learning_rate) == 1
    mlnn.lRate = [repmat(opts.learning_rate,1,mlnn.n-1)];
elseif numel(opts.learning_rate) == mlnn.n - 1
    mlnn.lRate = opts.learning_rate;
else
    error(['The setting of learning rate in architecture must be one or corresponding to number of layer-1.']);
end

mlnn.momentum = opts.momentum;
mlnn.sparsity=opts.sparsity;
mlnn.dropout = opts.dropout;
mlnn.testing = 0; % feedward process using dropout
mlnn.denoise = opts.denoise;
mlnn.normGrad = opts.normGrad;% constrain GRADIENTS to have unit norm
mlnn.maxNorm = opts.maxNorm;% constrain Norm of input WEIGHTS to be within a ball of radius
mlnn.sparseGain = opts.sparseGain;
mlnn.wPenalty = opts.wPenalty;

for u = 2 : mlnn.n
    % weights and weigth momentum
    range = sqrt(6/((mlnn.size(u) + mlnn.size(u-1))));
    mlnn.W{u-1} = (rand(mlnn.size(u),mlnn.size(u-1))-0.5)*2*range;
    mlnn.vW{u-1} = zeros(size(mlnn.W{u-1}));
    
    %bias and bias momentum
    mlnn.b{u-1} = zeros(mlnn.size(u),1);
    mlnn.vb{u-1} = zeros(size(mlnn.b{u-1}));
    
    %mean activations (for sparsity)
    mlnn.meanAct{u} = zeros(1,mlnn.size(u));
end

end