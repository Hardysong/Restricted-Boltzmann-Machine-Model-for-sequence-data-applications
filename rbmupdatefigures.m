function rbmupdatefigures(fhandle,L,opts,i)
%NNUPDATEFIGURES updates figures during training
if i > 1 %dont plot first point, its only a point   
    x_ax = 1:i;
    % create legend
    if opts.validation == 1
        M            = {'Training','Validation'};
    else
        M            = {'Training'};
    end
    
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
    xlabel('Number of epochs'); ylabel('Error');title('Error');
    legend(p, M,'Location','NorthEast');
    set(gca, 'Xlim',[0,opts.numepochs + 1])
    
    drawnow;
end
end