function [H, F] = AssembleMatrix(x, vdt, vertMass, edges, faces, edgeAdjFaces, edgeRestLen, isFixedVert, gravity, youngsModulus, lexuralModulus, timeStep)

    dim = length(x);
    numVerts = length(x) / 3;
    numEdges = length(edges);
    
    F = zeros(dim, 1);
    H = sparse(dim, dim);
    dt2 = timeStep * timeStep;
    
    for i = 1 : numEdges
        idx1 = 3 * edges(i,1) - 2 : 3 * edges(i,1);
        idx2 = 3 * edges(i,2) - 2 : 3 * edges(i,2);
        idx = [idx1, idx2];
        
        x1 = x(idx1);
        x2 = x(idx2);
        restLen = edgeRestLen(i);
        [H1, F1] = SpringHessianForce(x1, x2, youngsModulus, restLen);
        H(idx, idx) = H(idx, idx) + H1;
        F(idx) = F(idx) + F1;
        
        if (length(edgeAdjFaces{i}) == 2)
            vIdx3 = sum(faces(edgeAdjFaces{i}(1),:)) - sum(edges(i,:));
            vIdx4 = sum(faces(edgeAdjFaces{i}(2),:)) - sum(edges(i,:));
            idx3 = 3 * vIdx3 - 2 : 3 * vIdx3;
            idx4 = 3 * vIdx4 - 2 : 3 * vIdx4;
            idx = [idx1, idx2, idx3, idx4];
            
            [H2, F2] = DihedralBendingHessianForce(x1, x2, x(idx3), x(idx4), lexuralModulus, restLen);
            H(idx, idx) = H(idx, idx) + H2;
            F(idx) = F(idx) + F2;
        end
    end
    
    %%  Implicit euler diagonal term.
    for i = 1 : numVerts
        idx = 3 * i - 2 : 3 * i;
        if (isFixedVert(i) == 1)
            F(idx) = [0; 0; 0];
            H(idx, idx) = H(idx, idx) + eye(3) * 1e8;
        else
            massWeight =  vertMass(i) / dt2;
            F(3*i-1) = F(3*i-1) - vertMass(i) * gravity;
            H(idx, idx) = H(idx, idx) + eye(3) * massWeight;
            F(idx) = F(idx) + massWeight * vdt(idx);
        end
    end
end