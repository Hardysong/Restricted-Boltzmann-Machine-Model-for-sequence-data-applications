function [X4,Y4] = GenData(Ndata)
%   Generate test data for demo
%   
c1 = 0.5;
r1 = 0.4;
r2 = 0.3;

% generate enough data to filter
N = 20 * Ndata;
X = rand(N,1);
Y = rand(N,1);
X1 = X((X-c1).*(X-c1) + (Y-c1).*(Y-c1) < r1*r1);
Y1 = Y((X-c1).*(X-c1) + (Y-c1).*(Y-c1) < r1*r1);

X2 = X1((X1-c1).^2+(Y1-c1).^2 > r2.^2);
Y2 = Y1((X1-c1).^2+(Y1-c1).^2 > r2.^2);

X3 = X2(abs(X2-Y2) > 0.05);
Y3 = Y2(abs(X2-Y2) > 0.05);

X4 = zeros(Ndata,1);
Y4 = zeros(Ndata,1);

for i = 1 : Ndata
    if X3(i) - Y3(i) > 0.05
        X4(i) = X3(i)+0.08;
        Y4(i) = Y3(i) + 0.18;
    else
        X4(i) = X3(i) - 0.08;
        Y4(i) = Y3(i) - 0.18;
    end
end

end

