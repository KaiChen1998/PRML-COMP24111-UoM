function confu_matrix = confusion(predict, label, num_class)

% Input: 
%       'predict': predict label
%       'label': true label
%       'num_class': number of classes
% Output:         
%       'confu_matrix': cell array, every cell coresponding to one
%                       attribute value

confu_matrix = {};

for i = 1:num_class % one confusion matrix for each class
    confu_matrix{i} = zeros(2,2);
    index = find(label == i - 1);
    confu_matrix{i}(1,1) = sum(predict(index) == label(index));
    confu_matrix{i}(2,1) = size(index, 1) - confu_matrix{i}(1,1);
    index = find(predict == i - 1);
    confu_matrix{i}(1,2) = size(index, 1) - confu_matrix{i}(1,1);
    index = find(label ~= i - 1);
    confu_matrix{i}(2,2) = size(index, 1) - confu_matrix{i}(1,2);
end
         