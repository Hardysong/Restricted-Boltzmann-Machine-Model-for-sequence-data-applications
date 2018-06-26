function dbn = dbntrain( dbn,x,opts )
%   Train the continuous DBN layer by layer
%   
%   author: yibo Sun
%   date: 2017/09/15
dbstop if error;

n = numel(dbn.sizes) - 1;

switch lower(opts.class)
    case 'crbm'
        if isfield(dbn.crbm{1},'partially_supervised') && dbn.crbm{1}.partially_supervised==1
            switch lower(dbn.crbm{1}.partially_supervised_type)
                case 'regression'
                    % BP method
                    dbn.crbm{1} = crbmtrain_partially_surpervised(dbn.crbm{1},x,opts);
                case 'classification'
                    % CD method
                    dbn.crbm{1} = crbmtrain_supervised_classes(dbn.crbm{1},x,opts);
            end
        else
            dbn.crbm{1} = crbmtrain_standard(dbn.crbm{1},x,opts);
        end
        
        for i = 2 : n
            if isfield(dbn.crbm{i},'partially_supervised') && dbn.crbm{i}.partially_supervised==1
                x = crbmup(dbn.crbm{i-1},x);
                if ~isempty(opts.x_val)
                    t_val = crbmup(dbn.crbm{i - 1},opts.x_val);
                    opts.x_val = t_val;
                end
                switch lower(dbn.crbm{i}.partially_supervised_type)
                    case 'regression'
                        dbn.crbm{i} = crbmtrain_partially_surpervised(dbn.crbm{i},x,opts);
                    case 'classification'
                        dbn.crbm{i} = crbmtrain_supervised_classes(dbn.crbm{i},x,opts);
                end
            else
                x = crbmup(dbn.crbm{i-1},x);
                if ~isempty(opts.x_val)
                    t_val = crbmup(dbn.crbm{i - 1},opts.x_val);
                    opts.x_val = t_val;
                end
                dbn.crbm{i} = crbmtrain_standard(dbn.crbm{i},x,opts);
            end
            
        end
        
    case 'gbrbm'
        
    case 'bbrbm'
        
end