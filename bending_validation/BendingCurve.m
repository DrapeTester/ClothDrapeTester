function [x, y, s] = BendingCurve(theta0, rho_g_len3_invAlpha, numSamples)
    
    guessXc = 0.25;
    
    [x, y, s, delta] = NumericalIntegrate(theta0, rho_g_len3_invAlpha, numSamples, guessXc);
    
    step = 0.25;
    epslion = delta;
    
    while (epslion ~= 0)
        
        step = sign(delta) * abs(step) / 2.0;
        guessXc = guessXc + step;
        
        [x, y, s, newDelta] = NumericalIntegrate(theta0, rho_g_len3_invAlpha, numSamples, guessXc);
        
        epslion = newDelta - delta;
        delta = newDelta;
    end
end


function [x, y, s, delta] = NumericalIntegrate(theta0, rho_g_len3_invAlpha, numSamples, guessXc)
    
    s = linspace(0, 1, numSamples);
    
    ds = s(2) - s(1);
    x = zeros(1, length(s));
    y = zeros(1, length(s));
    K = guessXc * rho_g_len3_invAlpha;
    dKdx = rho_g_len3_invAlpha * (s(1) - 1);
    dydx = tan(theta0 * pi / 180.0);
    ddydxds = -K * (1 + dydx^2);
    
    for i = 1:length(s) - 1
        
        dx = ds / sqrt(1 + dydx^2);
        dy = sign(dydx) * ds * sqrt(1 - 1 / (1 + dydx^2));
        
        x(i + 1) = x(i) + dx;
        y(i + 1) = y(i) + dy;
        
        K = max(0, K + dKdx * dx);
        dydx = dydx + ddydxds * ds;
        dKdx = rho_g_len3_invAlpha * (s(i + 1) - 1);
        ddydxds = -K * (1 + dydx^2);
    end
    
    delta = sum(x * ds) - guessXc;
end