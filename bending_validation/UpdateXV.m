function [x, vdt] = UpdateXV(x, vdt, dx, isFixedVert)

    for i = 1 : length(isFixedVert)
        idx = 3*i - 2 : 3*i;
        if isFixedVert(i)
            vdt(idx) = [0;0;0];
        else
            x(idx) = x(idx) + dx(idx);
        end
    end
end