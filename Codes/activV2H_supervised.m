function H = activV2H_supervised( crbm, V,T,pattern)
%   Activate:
%           v2h: using visible to calculate the state of hidden and using
%           the target information
%  Author: Yibo Sun
%  2018/06/25

narginchk(3,4);

if nargin == 3
    pattern = 'train';
end

% set the low and high boundary layer of sigmoid function
thetaL = crbm.thetaL;
thetaH = crbm.thetaH;

num = size(V,1);
if exist('T','var') && ~isempty(T)
    V1 = V * crbm.W + T * crbm.U' + repmat(crbm.c',num,1);
else
    V1 = V * crbm.W + repmat(crbm.c',num,1);
end

H = V1 + crbm.sig * randn(size(V1));

A = repmat(crbm.Ahid,num,1);

H = sigmFun(thetaL,thetaH,A,H);

if crbm.dropout > 0 && strcmpi(pattern,'train')
    H = H .* crbm.dropOutMask;
end

end