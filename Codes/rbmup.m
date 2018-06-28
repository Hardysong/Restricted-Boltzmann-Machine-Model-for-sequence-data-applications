function x = rbmup( rbm,x,opts )
%   pass the x up to next hidden layer as input data for next crbm
%   此处显示详细说明
x = rbmV2H(rbm,x,opts,[],0,'inference');

end