function precise = object_acc(predict, label)

% Input:
%       "predict": prediction of every datapoint,(N,1)vector
%       "label": true label of every datapoint,(N,1)vector
% Output:
%       "precise": prediction accuracy of every object
%                  assume there are M object in label
%                  then precise will be a (1,M) vector
c = unique(label);
precise = zeros(1,length(c));
for i = 1:length(c)
    index = find(label == c(i));
    precise(i) = sum(predict(index) == label(index)) / size(index,1);
end