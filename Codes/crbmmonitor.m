function [ loss,crbm,str_perf ] = crbmmonitor(crbm,loss,train_x,opts,epoch)
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
%train_x(:,end) = [];

if mod(epoch,opts.test_interval) == 0
    % training performance
    [x_re,re_loss] = reconstruct(crbm,train_x,opts);
    loss.train.e(end + 1) = re_loss;
    if ~isempty(opts.x_val)
        [x_re,re_loss] = reconstruct(crbm,opts.x_val,opts);
        loss.val.e(end + 1)   = re_loss;
        crbm.testing = 1;
    else
        loss.val.e(end + 1) = NaN;
    end
    str_perf = sprintf('; Full-batch train mse = %f, val mse = %f, sigma = %f', loss.train.e(end), loss.val.e(end),crbm.sig);
    
end

end