function x = crbmup( crbm,x )
%   pass the x up to next hidden layer as input data for next crbm
%   �˴���ʾ��ϸ˵��
x = activV2H(crbm,x,'inference');

end

