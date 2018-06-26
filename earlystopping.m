function earlystop = earlystopping( model,opts,earlystop,loss,epoch )
%------------------------------------------------------
%   apply regulization of early stopping if enabled
%   This function tests wether the current performace is better than the
%   best performance.
%-------------------------------------------------------
% Yibo Sun

if opts.early_stopping &&  mod(epoch,opts.test_interval) == 0
    isbest = 0;
    % the performance information is from struct loss
    if  earlystop.best_err > loss.val.e(end)
        isbest = 1;
        err = loss.val.e(end);
    end
    
    if isbest
        earlystop.type = loss.type;
        earlystop.best_str = ' The current Best model ***';
        earlystop.best_err = err;
        earlystop.best_model = model;
        earlystop.best_eopch = epoch;
        earlystop.patience = opts.patience;
        
    else
        earlystop.best_str = ' ';
        earlystop.patience = earlystop.patience-1;
    end
end


end

