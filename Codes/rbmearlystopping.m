function earlystop = rbmearlystopping(rbm,opts,earlystop,loss,epoch)
%NNEARLYSTOP applies early stopping if enabled
%   Internal function.
%   If earlystopping is enabled the function tests wether the 
%   current performance is better than the best performance seen.
%   For classification RBM the validatio error is checked for non classification
%   RBM's the ratio of free energies is checked. 


if opts.early_stopping &&  mod(epoch,opts.test_interval) == 0
    isbest = 0;
    % for classification RBM's check if validatio is better than
    % current best
    if  earlystop.best_err > loss.val.e(end)
        isbest = 1;
        err = loss.val.e(end);
    end
    
    if isbest
        earlystop.type = loss.type;
        earlystop.best_str = ' ***';
        earlystop.best_err = err;
        earlystop.best_eopch = epoch;
        earlystop.patience = opts.patience;
        rbm.loss = loss;
        rbm.earlystop = earlystop;
        earlystop.best_rbm = rbm;
        
    else
        earlystop.best_str = ' ';
        earlystop.patience = earlystop.patience-1;
    end
end