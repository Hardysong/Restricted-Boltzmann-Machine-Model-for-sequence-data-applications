function [x,loss] = reconstruct( rbm,x,opts )
%   reconstruct the input data (x) and give the ereconstruct error (loss)
%   
%   author: yibo Sun
%   Date: 2017/09/16
narginchk(3,3);

ksteps = opts.CDk;

switch lower(opts.class)
    case 'crbm'
        v0 = x;
        h0 = activV2H(rbm,v0);
        
        % reconstruct phase
        v_state_k = v0;
        h_state_k = h0;
        for k = 1 : ksteps
            v_state_k = activH2V(rbm,h_state_k);
            h_state_k = activV2H(rbm,v_state_k);
        end
        
        x = v_state_k;
        
        %loss = sum(sum((v0-x).^2)) ./ size(x,1);
        loss = cost(x,v0,opts.costFun);
    case {'gbrbm','bbrbm'}
        v0 = x;
        [~,h0_sample] = rbmV2H(rbm,v0,opts,[],1,'inference');
        
        % reconstruct phase
        v_k = v0;
        h_state_k = h0_sample;
        for k = 1 : ksteps
            [v_k,~,~,~] = rbmH2V(rbm,h_state_k,opts,0);
            [~,h_state_k] = rbmV2H(rbm,v_k,opts);
        end
        
        x = v_k;
        
        loss = cost(x,v0,opts.costFun);
end

end


function J = cost(netOut,targets,costFun)
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
    case 'expll'
        % exponential log likelihood (poisson regression)
        J = sum(sum((netOut - targets.*log(netOut))))/nObs;
    case 'xent'
        % ½»²æìØÎó²î cross entropy (binary classification/Logistic regression)
        J = -sum(sum(targets.*log(netOut) + (1-targets).*log(1-netOut+0.0001)))/nObs;
    case 'mcxent'
        % multi-class cross entropy (classification) -> softmax
        J = -sum(sum(targets.*log(netOut)))./nObs;
    case {'cc','correlation'}
        J = corr2(netOut(:),targets(:));
end
end