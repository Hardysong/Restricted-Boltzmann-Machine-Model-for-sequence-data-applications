function mlnn = mlnn_updateParams(mlnn)
%-----------------------------------------------
%   Update netword parameters based on the gradient, perform regularization
%   such as weight decay and weight rescale
%-----------------------------------------------
%   Author: Yibo Sun

wPenalty = 0;
momentum = mlnn.momentum;
for lL = 1 : mlnn.n-1
    if mlnn.wPenalty > 0
        % L2-weight decay
        wPenalty = mlnn.W{lL} * mlnn.wPenalty;
    elseif mlnn.wPenalty < 0
        % (Approximate) L1-weight decay
        wPenalty = sign(mlnn.W{lL}) * mlnn.wPenalty;
    end
    
    % Update weight and bias (momentum)
    mlnn.vW{lL} = momentum * mlnn.vW{lL} + mlnn.lRate(lL)*(mlnn.dW{lL} + wPenalty);
    mlnn.vb{lL} = momentum * mlnn.vb{lL} + mlnn.lRate(lL)*mlnn.db{lL};
    
    % Update weights
    mlnn.W{lL} = mlnn.W{lL} - mlnn.vW{lL};
    mlnn.b{lL} = mlnn.b{lL} - mlnn.vb{lL};
    
    % Constrain norm of input weights to be within a Ball of radius
    % net.maxNorm
    if mlnn.maxNorm < Inf && lL == 1
        rescale = sqrt(sum([mlnn.W{lL},mlnn.b{lL}].^2,2));
        mask = rescale > mlnn.maxNorm;
        rescale(~mask) = 1;
        rescale(mask) = rescale(mask)/mlnn.maxNorm;
        
        mlnn.W{lL} = bsxfun(@rdivide,mlnn.W{lL},rescale);
        mlnn.b{lL} = mlnn.b{lL}./rescale;
    end

end

end

