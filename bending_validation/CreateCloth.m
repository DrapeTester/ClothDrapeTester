function cloth = CreateCloth(v, edges, faces, edgeAdjFaces, density)

cloth.totalArea = 0;
cloth.numVerts = size(v, 1);
cloth.numEdges = size(edges, 1);
cloth.numFaces = size(faces, 1);
cloth.vertMass = zeros(1, cloth.numVerts);
cloth.faceArea = zeros(1, cloth.numFaces);
cloth.edgeRestLen = zeros(1, cloth.numEdges);
cloth.edgeRestHeight = cell(1, cloth.numEdges);

for i = 1 : cloth.numFaces
    face = faces(i,:);
    A = v(face(1),:);
	B = v(face(2),:);
	C = v(face(3),:);
	cloth.faceArea(i) = norm(cross(B - A, C - B)) / 2;
    cloth.totalArea = cloth.totalArea + cloth.faceArea(i);
	cloth.vertMass(face(1)) = cloth.vertMass(face(1)) + density * cloth.faceArea(i) / 3;
    cloth.vertMass(face(2)) = cloth.vertMass(face(2)) + density * cloth.faceArea(i) / 3;
    cloth.vertMass(face(3)) = cloth.vertMass(face(3)) + density * cloth.faceArea(i) / 3;
end

for i = 1 : cloth.numEdges
    A = v(edges(i,1),:);
    B = v(edges(i,2),:);
    restLen = norm(A - B);
    cloth.edgeRestLen(i) = restLen;
    cloth.edgeRestHeight{i} = zeros(size(edgeAdjFaces{i}));
    for j = 1 : length(edgeAdjFaces{i})
        fIdx = edgeAdjFaces{i}(j);
        faceArea = cloth.faceArea(fIdx);
        cloth.edgeRestHeight{i}(j) = 2 * faceArea / restLen;
    end
end