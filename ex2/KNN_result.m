function [predict, accuracy] = KNN_result(k, train, y_tr, test, y_te)

% Input:
%       "k": k votes to get the class of a test point
%		"test": test set, a maxtrix with dimension of N, D 
%		"train": training set, a maxtrix with dimension of N',D
% 		"y_tr": true labels of train set (N',1)
%       "y_te": true labels of test set (N, 1)
%
% Output: 
%       predict: predict labels of test set
%       accuracy: accuracy on the test set

[N, D] = size(test); % N is the size of test set, and D is the dimension of datapoint
predict = zeros(N,1);
for i=1:N
	predict(i) = knearest(k, test(i,:), train, y_tr);
end
accuracy = mean(predict == y_te);

	
