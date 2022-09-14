clear; clc; clf;

L = 0.09;               % m
gravity = 9.8;          % m/s^2
density = 0.1;          % kg/m^2
timeStep = 1 / 50;      % s 
youngsModulus = 1;      % N/m
fileName = 'ribbon.obj';
numSamples = 50

modulus_lst = GenerateModulusForSampling(density, L, gravity, numSamples);
[master_x, master_y] = LoadMasterCurve();
ratio_threshold = 1e-3;

%subplot(1,2,1)
PlotMasterCurve(master_x, master_y);

% ------------------------------Ours---------------------------
for mod_id = 1:length(modulus_lst)
    cur_mod = modulus_lst(mod_id);
    mesh = ReadObj(fileName);
    % shift our mesh to the origin
    mesh.v(:,2) = mesh.v(:,2)  * 0;
    isFixedVert = SetFixed(mesh.v);
    meshTopo = BuildTopo(mesh.f.v, length(mesh.v));
    cloth = CreateCloth(mesh.v, meshTopo.edges, mesh.f.v, meshTopo.edgeAdjFaces, density);

    x = zeros(3 * length(mesh.v), 1);
    vdt = zeros(length(x), 1);
    for i = 1 : size(mesh.v, 1)
        x(3*i-2 : 3*i) = mesh.v(i, :)';
    end

    
    preratio = 0;
    dratio = 1e3;
    for i = 1 : 2000
        ratio =  GetHW(x);
        
        disp("cur iter " + i + " HW ratio " + ratio + " dratio " + dratio)
        [A, b] = AssembleMatrix(x, vdt, cloth.vertMass, meshTopo.edges, mesh.f.v, meshTopo.edgeAdjFaces, cloth.edgeRestLen, isFixedVert, gravity, youngsModulus, cur_mod, timeStep);
        dx = A \ b;
        [x, vdt] = UpdateXV(x, vdt, dx, isFixedVert);
%         subplot(1, 2, 2)
%         cla
%         trimesh(mesh.f.v, x(1:3:end), -x(3:3:end), x(2:3:end))
%         axis equal
%         drawnow
        
        dratio = abs(ratio - preratio);
        preratio = ratio;
        if dratio < ratio_threshold && i > 10
            break
        end
    end


    %subplot(1,2,1)
    hold on 
    
    rho_g_len3_invAlpha = density * gravity * L^3 / cur_mod;
    color = [68 / 255, 114 / 255, 196 / 255]
    scatter(rho_g_len3_invAlpha, ratio, 'filled', 'MarkerEdgeColor', color, 'MarkerFaceColor', color)

    drawnow
end


% ------------------------------Discrete Shell---------------------------
for mod_id = 1:length(modulus_lst)
    cur_mod = modulus_lst(mod_id);
    mesh = ReadObj(fileName);
    % shift our mesh to the origin
    mesh.v(:,2) = mesh.v(:,2)  * 0;
    isFixedVert = SetFixed(mesh.v);
    meshTopo = BuildTopo(mesh.f.v, length(mesh.v));
    cloth = CreateCloth(mesh.v, meshTopo.edges, mesh.f.v, meshTopo.edgeAdjFaces, density);

    x = zeros(3 * length(mesh.v), 1);
    vdt = zeros(length(x), 1);
    for i = 1 : size(mesh.v, 1)
        x(3*i-2 : 3*i) = mesh.v(i, :)';
    end

    
   
    preratio = 0;
    dratio = 1e3;
    for i = 1 : 2000
        ratio =  GetHW(x);
        
        disp("cur iter " + i + " HW ratio " + ratio + " dratio " + dratio)
        [A, b] = AssembleMatrix(x, vdt, cloth.vertMass, meshTopo.edges, mesh.f.v, meshTopo.edgeAdjFaces, cloth.edgeRestLen, isFixedVert, gravity, youngsModulus, cur_mod * 3, timeStep);
        dx = A \ b;
        [x, vdt] = UpdateXV(x, vdt, dx, isFixedVert);
%         subplot(1, 2, 2)
%         cla
%         trimesh(mesh.f.v, x(1:3:end), -x(3:3:end), x(2:3:end))
%         axis equal
%         drawnow
        
        dratio = abs(ratio - preratio);
        preratio = ratio;
        if dratio < ratio_threshold  && i > 10
            break
        end
    end


    %subplot(1,2,1)
    hold on 
    color = [237 /  255, 125/  255, 49/  255]
    rho_g_len3_invAlpha = density * gravity * L^3 / cur_mod;
    scatter(rho_g_len3_invAlpha, ratio, 'filled', 'MarkerEdgeColor', color, 'MarkerFaceColor', color)
    
    drawnow
end

% ------------------------------ARCSim---------------------------
dratio = 1e3
for mod_id = 1:length(modulus_lst)
    cur_mod = modulus_lst(mod_id);
    mesh = ReadObj(fileName);
    % shift our mesh to the origin
    mesh.v(:,2) = mesh.v(:,2)  * 0;
    isFixedVert = SetFixed(mesh.v);
    meshTopo = BuildTopo(mesh.f.v, length(mesh.v));
    cloth = CreateCloth(mesh.v, meshTopo.edges, mesh.f.v, meshTopo.edgeAdjFaces, density);

    x = zeros(3 * length(mesh.v), 1);
    vdt = zeros(length(x), 1);
    for i = 1 : size(mesh.v, 1)
        x(3*i-2 : 3*i) = mesh.v(i, :)';
    end

    
   
    preratio = 0;
    dratio = 1e3;
    for i = 1 : 2000
        ratio =  GetHW(x);
        
        disp("cur iter " + i + " HW ratio " + ratio + " dratio " + dratio)
        [A, b] = AssembleMatrix(x, vdt, cloth.vertMass, meshTopo.edges, mesh.f.v, meshTopo.edgeAdjFaces, cloth.edgeRestLen, isFixedVert, gravity, youngsModulus, cur_mod / 4, timeStep);
        dx = A \ b;
        [x, vdt] = UpdateXV(x, vdt, dx, isFixedVert);
%         subplot(1, 2, 2)
%         cla
%         trimesh(mesh.f.v, x(1:3:end), -x(3:3:end), x(2:3:end))
%         axis equal
%         drawnow
        
        dratio = abs(ratio - preratio);
        preratio = ratio;
        if dratio < ratio_threshold  && i > 10
            break
        end
    end


    %subplot(1,2,1)
    hold on 
    rho_g_len3_invAlpha = density * gravity * L^3 / cur_mod;
    color = [112 /  255, 173/  255, 71/  255]
    scatter(rho_g_len3_invAlpha, ratio, 'filled', 'MarkerEdgeColor',color, 'MarkerFaceColor',color)
    
    drawnow
end