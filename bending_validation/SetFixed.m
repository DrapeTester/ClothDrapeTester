function isFixedVert = SetFixed(v)
    
    isFixedVert = zeros(1, length(v));
    
    for i = 1 : length(v)
        if (v(i,1) < 0.002)
            isFixedVert(i) = 1;
        end
    end
end