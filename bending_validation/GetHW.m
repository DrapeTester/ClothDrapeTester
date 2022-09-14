function [ratio] = GetHW(pos)
    numOfVertices = length(pos) / 3;
    min_y = 1e3;
    max_x = -1e3;
    for i = 0:numOfVertices-1
        cur_pos = pos(3 * i + 1: 3 * i + 3);
        
        cur_y = cur_pos(2);
        cur_x = cur_pos(1);

        max_x = max(cur_x, max_x);
        min_y = min(cur_y, min_y);
    end
    ratio = abs(min_y) / abs(max_x);
end
