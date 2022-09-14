function meshTopo = BuildTopo(faces, numVerts)
%
%  fname      - file name
%  mesh       - output mesh
%  
%  mesh.v     - vertices positions
%  mesh.vt    - uv coordinates
%  mesh.vn    - normals
%  
%  mesh.f     - faces
%  mesh.f.v   - vertex indices for mesh.v 
%  mesh.f.vt  - vertex indices for mesh.vt 
%  mesh.f.vt  - vertex indices for mesh.vn
%

meshTopo.numEdges = 0;
meshTopo.numVerts = numVerts;
meshTopo.numFaces = size(faces,1);
meshTopo.vertAdjVerts = cell(numVerts, 1);
meshTopo.vertAdjEdges = cell(numVerts, 1);
meshTopo.vertAdjFaces = cell(numVerts, 1);
meshTopo.edgeAdjFaces = cell(0,1);
meshTopo.edges = zeros(0);
meshTopo.isBoundaryEdge = zeros(0);
meshTopo.isBoundaryVert = zeros(numVerts, 1);

% Build mesh topo.
for i = 1 : meshTopo.numFaces
	for j = 1 : 3
        v0 = faces(i, j);
        v1 = faces(i, mod(j, 3) + 1);
        n0 = length(meshTopo.vertAdjFaces{v0});
        meshTopo.vertAdjFaces{v0}(n0 + 1) = i;
        
        isFound = false;
        for k = 1 : length(meshTopo.vertAdjVerts{v0})
            if meshTopo.vertAdjVerts{v0}(k) == v1
                edgeIdx = meshTopo.vertAdjEdges{v0}(k);
                meshTopo.edgeAdjFaces{edgeIdx}(2) = i;
                isFound = true;
            end 
        end
        
        if ~isFound
            meshTopo.numEdges = meshTopo.numEdges + 1;
            meshTopo.edges(meshTopo.numEdges, 1) = v0;
            meshTopo.edges(meshTopo.numEdges, 2) = v1;
            num0 = length(meshTopo.vertAdjVerts{v0});
            num1 = length(meshTopo.vertAdjVerts{v1});
            meshTopo.vertAdjVerts{v0}(num0 + 1) = v1;
            meshTopo.vertAdjVerts{v1}(num1 + 1) = v0;
            meshTopo.vertAdjEdges{v0}(num0 + 1) = meshTopo.numEdges;
            meshTopo.vertAdjEdges{v1}(num1 + 1) = meshTopo.numEdges;
            meshTopo.edgeAdjFaces{meshTopo.numEdges}(1) = i;
        end
	end
end

% Find all boundary verts.
for i = 1 : meshTopo.numEdges
    if length(meshTopo.edgeAdjFaces{i}) < 2
        meshTopo.isBoundaryVert(meshTopo.edges(i, 1)) = true;
        meshTopo.isBoundaryVert(meshTopo.edges(i, 2)) = true;
        meshTopo.isBoundaryEdge(i) = true;
    end
end