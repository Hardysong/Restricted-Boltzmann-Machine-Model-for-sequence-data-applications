function H = activV2H( crbm, V,pattern)
%   Activate:
%           v2h: using visible to calculate the state of hidden
%           input:
%               crbm: struct
%               V: visible units
%               pattern: train or inference, train->dropout, inference->no 
%               dropout
%   Author; Yibo Sun
%   Date: 2018/06/25

narginchk(2,3);

if nargin == 2
    pattern = 'train';
end

% set the low and high boundary layer of sigmoid function
thetaL = crbm.thetaL;
thetaH = crbm.thetaH;

num = size(V,1);

V1 = V * crbm.W + repmat(crbm.c',num,1);

H = V1 + crbm.sig * randn(size(V1));

A = repmat(crbm.Ahid,num,1);

H = sigmFun(thetaL,thetaH,A,H);

if crbm.dropout > 0 && strcmpi(pattern,'train')
    H = H .* crbm.dropOutMask;
end

end

