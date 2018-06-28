function x = crbmup( crbm,x )
%   pass the x up to next hidden layer as input data for next crbm
%   此处显示详细说明
x = activV2H(crbm,x,'inference');

end

