function PlotMasterCurve(gamma, ratio)
    hold on
    set(gca, 'Box', 'on')
    set(gca, 'XGrid', 'on')
    set(gca, 'YGrid', 'on')
    set(gca, 'XScale', 'log')
    set(gca, 'YScale', 'log')
    plot(gamma, ratio, 'black')
    xlim([10^-3, 10^4])
    ylim([10^-4, 10^3])
    xlabel('\Gamma')
    ylabel('H / W')
end