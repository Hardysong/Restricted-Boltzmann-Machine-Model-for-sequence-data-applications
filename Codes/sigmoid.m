function P = sigmoid( in)
%   --------------------------------------------------------
%   Sigmoid activation function
%       input: 
%           in: pre activation value
%       output
%           P:activation value
%   Author: Yibo Sun
%   -------------------------------------------------------

P = 1./(1+exp(-in));

end

