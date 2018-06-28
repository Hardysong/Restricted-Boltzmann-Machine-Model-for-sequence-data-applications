function X = sigmFun( L,H,A,P )
%   sigmoid function with lower and upper asymptotes at L and H,
%   respectively. Parameter A controls the steepness of the sigmoid
%   function, and thus the nature of the unit's stochastic behaviour
%   
%    author: yibo Sun
%    Date:?20170915
narginchk(4,4);

x_in = A.*P;

X = L + (H - L).*(1./(1+exp(-x_in)));

end

