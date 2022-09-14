function mesh = ReadObj(fname)
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

vn = 1;
v_h = [];
vtn = 1;
vt_h = [];
vnn = 1;
vn_h = [];

fn = 1;
fv_h = [];
fvt_h = [];
fvn_h = [];

fid = fopen(fname);

while 1
    tline = fgetl(fid);
    if ~ischar(tline) % end of file
        break
    end  
    ln = sscanf(tline,'%s',1); % type
    switch ln
        case 'v'  
%             tline
            temp = sscanf(tline(2:end),'%f');
            if(vn>size(v_h,1))
                v_h  = double_array(v_h ,temp );
            else
                v_h (vn,:) = temp (:);
            end
            vn = vn + 1; 
        case 'vt'  
           temp = sscanf(tline(3:end),'%f') ;
            if(vtn>size(vt_h,1))
                vt_h  = double_array(vt_h ,temp );
            else
                vt_h (vtn,:) = temp (:);
            end
            vtn = vtn + 1; 
           
        case 'vn'  
            temp = sscanf(tline(3:end),'%f') ;
            if(vnn>size(vn_h,1))
                vn_h  = double_array(vn_h ,temp );
            else
                vn_h (vnn,:) = temp (:);
            end
            vnn = vnn + 1; 
            
        case 'f'   % face definition
%             ln
%             tline
            fv = []; fvt = []; fvn = [];
            str = tline(2:end);
            vertices_num = sum(str == ' ');
            nf = sum(str == '/') / vertices_num;
            nf = ceil(nf);
            switch(nf)
                case 0
                    C = textscan(str,'%d');
                    fv = C{1};
                case 1
                    C = textscan(str,'%d/%d');
                    fv = C{1};
                    fvt = C{2};
                case 2
                    C = textscan(str,'%d/%d/%d');
                    fv = C{1};
                    fvt = C{2};
                    fvn = C{3};
            end
           
            if(fn>size(fv_h,1))
                fv_h  = double_array(fv_h ,fv );
                fvt_h = double_array(fvt_h,fvt);
                fvn_h = double_array(fvn_h,fvn);
            else
                fv_h (fn,:) = fv (:);
                fvt_h(fn,:) = fvt(:);
                fvn_h(fn,:) = fvn(:);
            end
          
            fn = fn + 1 ;
        otherwise
                % do nothing
    end
end
fclose(fid);

f.v  = shrink_to_fit(fv_h,fn);
f.vt = shrink_to_fit(fvt_h,fn);
f.vn = shrink_to_fit(fvn_h,fn);
% set up matlab object

mesh.v = shrink_to_fit(v_h,vn); 
mesh.vt = shrink_to_fit(vt_h,vtn); 
mesh.vn = shrink_to_fit(vn_h,vnn); 
mesh.f = f;
end

function r = shrink_to_fit(array,nv)
    r = array(1:nv - 1,:);
end

function r = double_array(array,vab)
    sz = size(array,1);
    r = zeros(sz * 2,length(vab));
    r(1:sz,:) = array(1:sz,:);
    r(sz + 1,:) = vab(:);
end