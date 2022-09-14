function [master_x, master_y] = LoadMasterCurve(pos)
    data  = load("master_curve_data");
    master_x = data.master_x;
    master_y = data.master_y;
end