function [modulus_lst] = GenerateModulusForSampling(rho, L, g, samples)
    log10_gamma_st = -2;
    log10_gamma_ed = 2;
    
    log10_gamma_lst = linspace(log10_gamma_st, log10_gamma_ed, samples);
    gamma_lst = 10.^(log10_gamma_lst);
    modulus_lst = rho * g * L^3 ./ gamma_lst;
    
end
