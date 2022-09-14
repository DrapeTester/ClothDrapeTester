function [H, F] = SpringHessianForce(x1, x2, youngsModulus, restLen)

    currLen = norm(x1 - x2);
    dir = (x1 - x2) / currLen;
    k = youngsModulus / restLen;
    force = k * (currLen - restLen) * dir;
    
    hessian = k * (dir * dir');
    if (currLen > restLen)
        diagVal = k * (currLen - restLen) / currLen;
        hessian(1,1) = hessian(1,1) + diagVal;
        hessian(2,2) = hessian(2,2) + diagVal;
        hessian(3,3) = hessian(3,3) + diagVal;
    end
    
    F = [ -force; force ];
    H = [ hessian, -hessian;
         -hessian,  hessian ];
end