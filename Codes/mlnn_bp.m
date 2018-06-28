function mlnn = mlnn_bp( mlnn )
%--------------------------------------------------
%   Perform gradient descent on the loss w.r.t each of the model parameters
%   using the backpropagation algorithm. Return updated netword object. <mlnn>
%   Author: Yibo Sun
%--------------------------------------------------
sparsityError = 0;

% derivative of output activation function
dAct = calcActDerive(mlnn.act{end},mlnn.actFun{end});

% Error partial derivatives
dE{mlnn.n} = mlnn.netError.*dAct;

% Backpropagate error algorithm
for lL = (mlnn.n - 1) : -1 : 2
    
    if strcmpi(mlnn.actFun{lL-1},'sigmoid') && mlnn.sparsity > 0 && mlnn.epoch > 1
        KL = -mlnn.sparsity./mlnn.meanAct{lL} + (1-mlnn.sparsity)./(1-mlnn.meanAct{lL});
        sparsityError = mlnn.sparseGain*mlnn.lRate(lL).*KL;
    end
    
    % Layer error contribution % Error back-propagation
    propError = dE{lL+1}*mlnn.W{lL};
    
    % Derivative of Activation Function
    dAct = calcActDerive(mlnn.act{lL},mlnn.actFun{lL-1});
    
    % Calculate layer error signal (include sparsity)
    dE{lL} = bsxfun(@plus,propError,sparsityError).*dAct;
    if mlnn.dropout > 0 % add by yibo Sun. 
        dE{lL} = dE{lL} .* mlnn.dropOutMask{lL};
    end

end

% Calculate dE/dW for each layer
for lL = 1 : mlnn.n - 1
    mlnn.dW{lL} = (dE{lL+1}'*mlnn.act{lL})/size(dE{lL+1},1);
    mlnn.db{lL} = sum(dE{lL+1})'/size(dE{lL+1},1);
    if mlnn.normGrad
        % Constrain gradients to have unit norm 
        normdW = norm([mlnn.dW{lL}(:);mlnn.db{lL}(:)]);
        mlnn.dW{lL} = mlnn.dW{lL}/normdW;
        mlnn.db{lL} = mlnn.db{lL}/normdW;
    end
end

end

function dAct = calcActDerive(in,actFun)
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