function [H, F] = DihedralBendingHessianForce(x1, x2, x3, x4, lexuralModulus, restLen)

    x12 = x2 - x1;
    x13 = x3 - x1;
    x14 = x4 - x1;
    n1 = cross(x12, x13);
    n2 = cross(x14, x12);
    
    currLen = norm(x12);
    h1 = norm(n1) / currLen;
    h2 = norm(n2) / currLen;
    
    e = x12 / currLen;
    n1 = n1 / norm(n1);
    n2 = n2 / norm(n2);
    w1 = dot(e, x13) / currLen;
    w2 = dot(e, x14) / currLen;
    
    dAdx = [(w1 - 1) * n1 / h1 + (w2 - 1) * n2 / h2;
           -(w1 * n1 / h1 + w2 * n2 / h2);
            n1 / h1;
            n2 / h2];
    
    angle = DihedralAngle(e, n1, n2);
    kappa = SignedCurveture(angle, h1, h2);
    bendingRigidity = lexuralModulus * restLen;
    
    F = bendingRigidity *      kappa      * dAdx;     
    H = bendingRigidity * (2 / (h1 + h2)) * (dAdx * dAdx');
end

function kappa = SignedCurveture(angle, h1, h2)
    kappa = 2 * angle / (h1 + h2);
end

function angle = DihedralAngle(e, n1, n2)
    cosAngle = dot(n1, n2);
    sinAngle = dot(e, cross(n1, n2));
    angle = atan2(sinAngle, cosAngle);
end