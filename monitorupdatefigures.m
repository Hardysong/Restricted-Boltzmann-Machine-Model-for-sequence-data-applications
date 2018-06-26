function monitorupdatefigures(fhandle,L,opts,i)
%---------------------------------------------------------
%   update figures of train error and/or validation error durning training 
%---------------------------------------------------------
% Yibo Sun

if i > 1 %dont plot first point, its only a point   
    x_ax = 1:i;
    % create legend
    if opts.validation == 1
        M            = {'Training','Validation'};
    else
        M            = {'Training'};
    end
    if isempty(L.train.e_frac)
        %create data for plots
        plot_x       = x_ax';
        plot_ye      = L.train.e';
        
        %add error on validation data if present
        if opts.validation == 1
            plot_x       = [plot_x, x_ax'];
            plot_ye      = [plot_ye,L.val.e'];
        end
        
        %    plotting
        figure(fhandle);
        
        p = plot(plot_x,plot_ye);
        xlabel('Number of epochs'); ylabel(['Error (' L.type ')']);title('Error performance');
        legend(p, M,'Location','NorthEast');
        set(gca, 'Xlim',[0,opts.numepochs + 1])
        
        drawnow;
    else
        %create data for plots
        plot_x       = x_ax';
        plot_ye      = L.train.e_frac';
        
        %add error on validation data if present
        if opts.validation == 1
            plot_x       = [plot_x, x_ax'];
            plot_ye      = [plot_ye,L.val.e_frac'];
        end
        
        %    plotting
        figure(fhandle);
        
        p = plot(plot_x,plot_ye);
        xlabel('Number of epochs'); ylabel('Error (Misclassification Rate)');title('Error performance');
        legend(p, M,'Location','NorthEast');
        set(gca, 'Xlim',[0,opts.numepochs + 1])
        
        drawnow;
    end
end


end

