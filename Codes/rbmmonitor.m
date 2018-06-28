function [ loss,rbm,str_perf ] = rbmmonitor(rbm,loss,train_x,opts,epoch)
% monitor Continuous RBM performance
%
% INPUTS:
%          crbm : a crbm struct
%           train_x : the original train data
%        opts : opts struct
%     epoch   : current epoch number
%
% Returns a updated loss struct
assert(nargin == 5, 'Wrong number of arguments');

if mod(epoch,opts.test_interval) == 0
    % training performance
    [x_re,re_loss] = reconstruct(rbm,train_x,opts);
    loss.train.e(end + 1) = re_loss;
    if ~isempty(opts.x_val)
        [x_re,re_loss] = reconstruct(rbm,opts.x_val,opts);
        loss.val.e(end + 1)   = re_loss;
        rbm.testing = 1;
    else
        loss.val.e(end + 1) = NaN;
    end
    str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
    
end

end