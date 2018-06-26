function loss = nnmonitor( mlnn,loss,train_x,train_y,val_x,val_y )
%   ------------------------------------------------------------
%   Evaluate the performance of multi-layer neural network
%   Returns a updated loss struct
%   ------------------------------------------------------------
%   Author: Yibo Sun
assert(nargin == 4 | nargin == 6,[mfilename ': Wrong number of argument']);

mlnn.testing = 1;

% training peformance
[ ~,L ] = mlnn_fp( mlnn,train_x,train_y );
loss.train.e(end + 1) = L;

% validation performance
if nargin == 6
    [~,L] = mlnn_fp(mlnn,val_x,val_y);
    loss.val.e(end+1) = L;
end
mlnn.testing = 0;

% Calcuate misclassification rate if softmax
if strcmpi(mlnn.actFun{end},'softmax')
    [er_train,dummy] = nnclassification_monitor(mlnn,train_x,train_y);
    loss.train.e_frac(end + 1) = er_train;
    
    if nargin == 6
        [er_val,dummy] = nnclassification_monitor(mlnn,val_x,val_y);
        loss.val.e_frac(end+1) = er_val;
    end
end

end

function [er,bad] = nnclassification_monitor(mlnn,x,y)
    mlnn = mlnn_fp( mlnn,x);
    [~,i] = max(mlnn.act{end},[],2);
    labels=i;
    [~,expected] = max(y,[],2);
    bad = find(labels~=expected);
    er = numel(bad)./size(x,1);
end