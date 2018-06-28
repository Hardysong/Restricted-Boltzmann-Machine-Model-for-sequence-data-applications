function [H,H_sample] = rbmV2H( rbm,v,opts,y,sample,pattern)
%-----------------------------------------------------
%   Calculate p(h=1/v) for the RBM (GB/BB RBM)
%   input:
%       rbm: struct
%       v: the activation of the visible layer
%       y: possible target variables (optional)
%       sample: Flag indicating whether to sample the states of the hidden
%       units
%       pattern, train or inference.
%   output:
%       H: activative of the hidden layer
%       H_sample: sample result of the H
%
%   Author: Yibo Sun
%-----------------------------------------------------
narginchk(3,6);
switch nargin
    case 3
        y = [];
        sample = 1;
        pattern = 'train';
    case 4
        sample = 1;
        pattern = 'train';
    case 5
        pattern = 'train';
end

H=[];
H_sample=[];

if ~isempty(y)
    switch lower(opts.class)
        case 'bbrbm'
            act_hid = bsxfun(@plus,v*rbm.W,rbm.c') + y * rbm.U';
        case 'gbrbm'
            % for compute stable, I did not use the square of rbm.sig
            V_hid = bsxfun(@rdivide,v,rbm.sig);
            act_hid = bsxfun(@plus,V_hid*rbm.W,rbm.c') + y * rbm.U';
    end
else
    switch lower(opts.class)
        case 'bbrbm'
            act_hid = bsxfun(@plus,v*rbm.W,rbm.c');
        case 'gbrbm'
            % for compute stable, I did not use the square of rbm.sig
            V_hid = bsxfun(@rdivide,v,rbm.sig);
            act_hid = bsxfun(@plus,V_hid*rbm.W,rbm.c');
    end
end
H = sigmoid(act_hid);
if sample
    H_sample = H > rand(size(H));
else
    H_sample = [];
end

if rbm.dropout > 0 && strcmpi(pattern,'train')
    H_sample= H_sample .* crbm.dropOutMask;
end

end

