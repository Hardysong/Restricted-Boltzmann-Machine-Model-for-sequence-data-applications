function dbn = dbnsetup( sizes, x, opts )
%   setup the structe of the deep belief networks according to the class
%   from opts
%   the dbn is consisted by stacked rbm 
%   
%   author: yibo Sun
%   2018/06/21

n = size(x,2); %line is the number of input data dim
dbn.sizes = [n,sizes];
DBN_class = opts.class;

switch lower(DBN_class)
    case 'crbm'
    %for crbm 
    for u = 1 : numel(dbn.sizes) - 1
        dbn.crbm{u}.lr_w = opts.crbm.learning_rate_W;
        dbn.crbm{u}.lr_a = opts.crbm.learning_rate_A;
        dbn.crbm{u}.momentum = opts.momentum;
        dbn.crbm{u}.sig = opts.sig;
        dbn.crbm{u}.weightcost = opts.wPenalty;
        dbn.crbm{u}.thetaL = opts.crbm.thetaL;% low boundary of sigmoid function
        dbn.crbm{u}.thetaH = opts.crbm.thetaH;% high boundary of sigmoid function
        dbn.crbm{u}.dropout = opts.dropout;
        % initialize the weight matrix
        % Note: initial the weight using gaussion distribute due to add
        % Gaussion noise to neural
        switch lower(opts.init_type)
            case 'gauss'
                dbn.crbm{u}.W = 0.1 * randn(dbn.sizes(u),dbn.sizes(u+1));
                dbn.crbm{u}.vW = 0.001 * randn(dbn.sizes(u),dbn.sizes(u+1));
            case 'uniform'
                range = sqrt(6/((dbn.size(u) + dbn.size(u+1))));
                dbn.crbm{u}.W = (rand(dbn.size(u),dbn.size(u+1))-0.5)*2*range;
                dbn.crbm{u}.vW = zeros(size(dbn.crbm{u}.W));
            otherwise
                error([mfilename ': init_type in opts should be either gauss or uniform']);
        end
        % bias of visible layer
        dbn.crbm{u}.b = zeros(dbn.sizes(u),1);
        dbn.crbm{u}.vb = zeros(dbn.sizes(u),1);
        % bias of hidden layers
        dbn.crbm{u}.c = zeros(dbn.sizes(u+1),1);
        dbn.crbm{u}.vc = zeros(dbn.sizes(u+1),1);
        
        % initial the parameter a_j that controls the steepness of the sigmoid
        % function
        dbn.crbm{u}.Avis = opts.crbm.Avis * ones(1,dbn.sizes(u));%0.1
        dbn.crbm{u}.Ahid = opts.crbm.Ahid * ones(1,dbn.sizes(u+1));
        dbn.crbm{u}.dA = zeros(1,dbn.sizes(u));
        
        if ~isempty(opts.y_train) && opts.partially_supervised
            % 
            if isvector(opts.partially_supervisedlayer)
                if length(opts.partially_supervisedlayer) == 1
                    if opts.partially_supervisedlayer == u
                        dbn.crbm{u}.partially_supervised = 1;
                        % classification method is reference 'A fast
                        % learning algorithm for deep belief nets' from
                        % Hinton 
                        % regression method is reference 'Greedy Layer-Wise
                        % Training of Deep Networks' from Bengio 2007
                        dbn.crbm{u}.partially_supervised_type = opts.partially_supervised_type;
                        dbn.crbm{u}.outputFun = opts.partially_supervised_outputFun;
                        dbn.crbm{u}.supervised_costFun = opts.partially_supervised_costFun;
                        dbn.crbm{u}.normGrad = opts.partially_supervised_normGrad;
                        dbn.crbm{u}.maxNorm = opts.partially_supervised_maxNorm;
                        N_output = size(opts.y_train,2);
                        switch lower(opts.init_type)
                            case 'gauss'
                                dbn.crbm{u}.U = 0.1 * randn(dbn.sizes(u+1),N_output);
                                dbn.crbm{u}.vU = 0.001 * zeros(dbn.sizes(u+1),N_output);
                            case 'uniform'
                                range = sqrt(6/((dbn.size(u+1) + N_output)));
                                dbn.crbm{u}.U = (rand(dbn.size(u+1),N_output)-0.5)*2*range;
                                dbn.crbm{u}.vU = zeros(size(dbn.crbm{u}.U));
                            otherwise
                                error([mfilename ': init_type in opts should be either gauss or uniform']);
                        end
                        % bias of target layer
                        dbn.crbm{u}.d = zeros(N_output,1);
                        dbn.crbm{u}.vd = zeros(N_output,1);
                    else
                        continue;
                    end
                elseif length(opts.partially_supervisedlayer) > 1
                    for l =1 : length(opts.partially_supervisedlayer)
                        if opts.partially_supervisedlayer(l) == u
                            dbn.crbm{u}.partially_supervised = 1;
                            % classification method is reference 'A fast
                            % learning algorithm for deep belief nets' from
                            % Hinton
                            % regression method is reference 'Greedy Layer-Wise
                            % Training of Deep Networks' from Bengio 2007
                            dbn.crbm{u}.partially_supervised_type = opts.partially_supervised_type;
                            dbn.crbm{u}.outputFun = opts.partially_supervised_outputFun;
                            dbn.crbm{u}.supervised_costFun = opts.partially_supervised_costFun;
                            dbn.crbm{u}.normGrad = opts.partially_supervised_normGrad;
                            dbn.crbm{u}.maxNorm = opts.partially_supervised_maxNorm;
                            N_output = size(opts.y_train,2);
                            switch lower(opts.init_type)
                                case 'gauss'
                                    dbn.crbm{u}.U = 0.1 * randn(dbn.sizes(u+1),N_output);
                                    dbn.crbm{u}.vU = 0.001 * zeros(dbn.sizes(u+1),N_output);
                                case 'uniform'
                                    range = sqrt(6/((dbn.size(u+1) + N_output)));
                                    dbn.crbm{u}.U = (rand(dbn.size(u+1),N_output)-0.5)*2*range;
                                    dbn.crbm{u}.vU = zeros(size(dbn.crbm{u}.U));
                                otherwise
                                    error([mfilename ': init_type in opts should be either gauss or uniform']);
                            end
                            % bias of target layer
                            dbn.crbm{u}.d = zeros(N_output,1);
                            dbn.crbm{u}.vd = zeros(N_output,1);
                        else
                            continue;
                        end
                    end
                end
            else
                error([mfilename ': the definition of opts.partially_supervised must be a vector']);
            end
        end
        
    end
case 'gbrbm'
    % for gbrbm 
    for u = 1 : numel(dbn.sizes) - 1
        dbn.gbrbm{u}.lr_w = opts.learning_rate;
        dbn.gbrbm{u}.momentum = opts.momentum;
        dbn.gbrbm{u}.sig = opts.sig;
        dbn.gbrbm{u}.weightcost = opts.wPenalty;
        dbn.gbrbm{u}.dropout = opts.dropout;
        % initialize the weight matrix
        switch lower(opts.init_type)
            case 'gauss'
                dbn.gbrbm{u}.W = 0.01 * randn(dbn.sizes(u),dbn.sizes(u+1));
                dbn.gbrbm{u}.vW = 0.001 * randn(dbn.sizes(u),dbn.sizes(u+1));
            case 'uniform'
                range = sqrt(6/((dbn.size(u) + dbn.size(u+1))));
                dbn.gbrbm{u}.W = (rand(dbn.size(u),dbn.size(u+1))-0.5)*2*range;
                dbn.gbrbm{u}.vW = zeros(size(dbn.crbm{u}.W));
            otherwise
                error([mfilename ': init_type in opts should be either gauss or uniform']);
        end
        % bias of visible layer
        dbn.gbrbm{u}.b = zeros(dbn.sizes(u),1);
        dbn.gbrbm{u}.vb = zeros(dbn.sizes(u),1);
        % bias of hidden layers
        dbn.gbrbm{u}.c = zeros(dbn.sizes(u+1),1);
        dbn.gbrbm{u}.vc = zeros(dbn.sizes(u+1),1);
        
        if ~isempty(opts.y_train) && opts.partially_supervised
            %
            if isvector(opts.partially_supervisedlayer)
                if length(opts.partially_supervisedlayer) == 1
                    if opts.partially_supervisedlayer == u
                        dbn.gbrbm{u}.partially_supervised = 1;
                        % classification method is reference 'A fast
                        % learning algorithm for deep belief nets' from
                        % Hinton
                        % regression method is reference 'Greedy Layer-Wise
                        % Training of Deep Networks' from Bengio 2007
                        dbn.gbrbm{u}.partially_supervised_type = opts.partially_supervised_type;
                        dbn.gbrbm{u}.outputFun = opts.partially_supervised_outputFun;
                        dbn.gbrbm{u}.supervised_costFun = opts.partially_supervised_costFun;
                        dbn.gbrbm{u}.normGrad = opts.partially_supervised_normGrad;
                        dbn.gbrbm{u}.maxNorm = opts.partially_supervised_maxNorm;
                        N_output = size(opts.y_train,2);
                        switch lower(opts.init_type)
                            case 'gauss'
                                dbn.gbrbm{u}.U = 0.1 * randn(dbn.sizes(u+1),N_output);
                                dbn.gbrbm{u}.vU = 0.001 * zeros(dbn.sizes(u+1),N_output);
                            case 'uniform'
                                range = sqrt(6/((dbn.size(u+1) + N_output)));
                                dbn.gbrbm{u}.U = (rand(dbn.size(u+1),N_output)-0.5)*2*range;
                                dbn.gbrbm{u}.vU = zeros(size(dbn.gbrbm{u}.U));
                            otherwise
                                error([mfilename ': init_type in opts should be either gauss or uniform']);
                        end
                        % bias of target layer
                        dbn.gbrbm{u}.d = zeros(N_output,1);
                        dbn.gbrbm{u}.vd = zeros(N_output,1);
                    else
                        continue;
                    end
                elseif length(opts.partially_supervisedlayer) > 1
                    for l =1 : length(opts.partially_supervisedlayer)
                        if opts.partially_supervisedlayer(l) == u
                            dbn.gbrbm{u}.partially_supervised = 1;
                            % classification method is reference 'A fast
                            % learning algorithm for deep belief nets' from
                            % Hinton
                            % regression method is reference 'Greedy Layer-Wise
                            % Training of Deep Networks' from Bengio 2007
                            dbn.gbrbm{u}.partially_supervised_type = opts.partially_supervised_type;
                            dbn.gbrbm{u}.outputFun = opts.partially_supervised_outputFun;
                            dbn.gbrbm{u}.supervised_costFun = opts.partially_supervised_costFun;
                            dbn.gbrbm{u}.normGrad = opts.partially_supervised_normGrad;
                            dbn.gbrbm{u}.maxNorm = opts.partially_supervised_maxNorm;
                            N_output = size(opts.y_train,2);
                            switch lower(opts.init_type)
                                case 'gauss'
                                    dbn.gbrbm{u}.U = 0.1 * randn(dbn.sizes(u+1),N_output);
                                    dbn.gbrbm{u}.vU = 0.001 * zeros(dbn.sizes(u+1),N_output);
                                case 'uniform'
                                    range = sqrt(6/((dbn.size(u+1) + N_output)));
                                    dbn.gbrbm{u}.U = (rand(dbn.size(u+1),N_output)-0.5)*2*range;
                                    dbn.gbrbm{u}.vU = zeros(size(dbn.gbrbm{u}.U));
                                otherwise
                                    error([mfilename ': init_type in opts should be either gauss or uniform']);
                            end
                            % bias of target layer
                            dbn.gbrbm{u}.d = zeros(N_output,1);
                            dbn.gbrbm{u}.vd = zeros(N_output,1);
                        else
                            continue;
                        end
                    end
                end
            else
                error([mfilename ': the definition of opts.partially_supervised must be a vector']);
            end
        end
    end
case 'bbrbm'
    % for bbrbm 
    for u = 1 : numel(dbn.sizes) - 1
        dbn.bbrbm{u}.lr_w = opts.learning_rate;
        dbn.bbrbm{u}.momentum = opts.momentum;
        dbn.bbrbm{u}.weightcost = opts.wPenalty;
        dbn.bbrbm{u}.dropout = opts.dropout;
        % initialize the weight matrix
        % initialize the weight matrix
        switch lower(opts.init_type)
            case 'gauss'
                dbn.bbrbm{u}.W = 0.01 * randn(dbn.sizes(u),dbn.sizes(u+1));
                dbn.bbrbm{u}.vW = 0.001 * randn(dbn.sizes(u),dbn.sizes(u+1));
            case 'uniform'
                range = sqrt(6/((dbn.size(u) + dbn.size(u+1))));
                dbn.bbrbm{u}.W = (rand(dbn.size(u),dbn.size(u+1))-0.5)*2*range;
                dbn.bbrbm{u}.vW = zeros(size(dbn.crbm{u}.W));
            otherwise
                error([mfilename ': init_type in opts should be either gauss or uniform']);
        end
        % bias of visible layer
        dbn.bbrbm{u}.b = zeros(dbn.sizes(u),1);
        dbn.bbrbm{u}.vb = zeros(dbn.sizes(u),1);
        % bias of hidden layers
        dbn.bbrbm{u}.c = zeros(dbn.sizes(u+1),1);
        dbn.bbrbm{u}.vc = zeros(dbn.sizes(u+1),1);
        
        if ~isempty(opts.y_train) && opts.partially_supervised
            % 
            if isvector(opts.partially_supervisedlayer)
                if length(opts.partially_supervisedlayer) == 1
                    if opts.partially_supervisedlayer == u
                        dbn.bbrbm{u}.partially_supervised = 1;
                        % classification method is reference 'A fast
                        % learning algorithm for deep belief nets' from
                        % Hinton 
                        % regression method is reference 'Greedy Layer-Wise
                        % Training of Deep Networks' from Bengio 2007
                        dbn.bbrbm{u}.partially_supervised_type = opts.partially_supervised_type;
                        dbn.bbrbm{u}.outputFun = opts.partially_supervised_outputFun;
                        dbn.bbrbm{u}.supervised_costFun = opts.partially_supervised_costFun;
                        dbn.bbrbm{u}.normGrad = opts.partially_supervised_normGrad;
                        dbn.bbrbm{u}.maxNorm = opts.partially_supervised_maxNorm;
                        N_output = size(opts.y_train,2);
                        switch lower(opts.init_type)
                            case 'gauss'
                                dbn.bbrbm{u}.U = 0.1 * randn(dbn.sizes(u+1),N_output);
                                dbn.bbrbm{u}.vU = 0.001 * zeros(dbn.sizes(u+1),N_output);
                            case 'uniform'
                                range = sqrt(6/((dbn.size(u+1) + N_output)));
                                dbn.bbrbm{u}.U = (rand(dbn.size(u+1),N_output)-0.5)*2*range;
                                dbn.bbrbm{u}.vU = zeros(size(dbn.bbrbm{u}.U));
                            otherwise
                                error([mfilename ': init_type in opts should be either gauss or uniform']);
                        end
                        % bias of target layer
                        dbn.bbrbm{u}.d = zeros(N_output,1);
                        dbn.bbrbm{u}.vd = zeros(N_output,1);
                    else
                        continue;
                    end
                elseif length(opts.partially_supervisedlayer) > 1
                    for l =1 : length(opts.partially_supervisedlayer)
                        if opts.partially_supervisedlayer(l) == u
                            dbn.bbrbm{u}.partially_supervised = 1;
                            % classification method is reference 'A fast
                            % learning algorithm for deep belief nets' from
                            % Hinton
                            % regression method is reference 'Greedy Layer-Wise
                            % Training of Deep Networks' from Bengio 2007
                            dbn.bbrbm{u}.partially_supervised_type = opts.partially_supervised_type;
                            dbn.bbrbm{u}.outputFun = opts.partially_supervised_outputFun;
                            dbn.bbrbm{u}.supervised_costFun = opts.partially_supervised_costFun;
                            dbn.bbrbm{u}.normGrad = opts.partially_supervised_normGrad;
                            dbn.bbrbm{u}.maxNorm = opts.partially_supervised_maxNorm;
                            N_output = size(opts.y_train,2);
                            switch lower(opts.init_type)
                                case 'gauss'
                                    dbn.bbrbm{u}.U = 0.1 * randn(dbn.sizes(u+1),N_output);
                                    dbn.bbrbm{u}.vU = 0.001 * zeros(dbn.sizes(u+1),N_output);
                                case 'uniform'
                                    range = sqrt(6/((dbn.size(u+1) + N_output)));
                                    dbn.bbrbm{u}.U = (rand(dbn.size(u+1),N_output)-0.5)*2*range;
                                    dbn.bbrbm{u}.vU = zeros(size(dbn.bbrbm{u}.U));
                                otherwise
                                    error([mfilename ': init_type in opts should be either gauss or uniform']);
                            end
                            % bias of target layer
                            dbn.bbrbm{u}.d = zeros(N_output,1);
                            dbn.bbrbm{u}.vd = zeros(N_output,1);
                        else
                            continue;
                        end
                    end
                end
            else
                error([mfilename ': the definition of opts.partially_supervised must be a vector']);
            end
        end
    end
    
end