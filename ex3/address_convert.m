function [c, d] = address_convert(index, num_class)

% Input: matrix 1-dim index
% Output: matrix 2-dim index

c = zeros(size(index,1), 1);
d = zeros(size(index,1), 1);

for i = 1:size(index, 1)
    c(i) = rem(index(i), num_class);
    if (c(i) == 0)
        c(i) = num_class;
    end
    
    d(i) = (index(i) - c(i)) / num_class + 1;
end