function x = rbmup( rbm,x,opts )
%   pass the x up to next hidden layer as input data for next crbm
%   �˴���ʾ��ϸ˵��
x = rbmV2H(rbm,x,opts,[],0,'inference');

end