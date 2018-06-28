function V = activH2V( crbm, H)
%   Activate:
%           h2v: using hidden to calculate the state of visible 
narginchk(2,2);

% set the low and high boundary layer of sigmoid function
thetaL = crbm.thetaL;
thetaH = crbm.thetaH;

num = size(H,1);

H1 = H * crbm.W' + repmat(crbm.b',num,1);

V = H1 + crbm.sig * randn(size(H1));

A = repmat(crbm.Avis,num,1);

V = sigmFun(thetaL,thetaH,A,V);

end

