function [predict, accuracy, confu_matrix] = NBTest(testAttributeSet, validLabel, p_condition, p_class, config)

% Input: 
%       'testAttributeSet', 'validLabel': Test set
%       'p_condition', 'p_class': Probability used for naive bayes
%       'config': Parameters useful for testing
% Output:
%       'predict': predict result
%       'accuracy': predict accuracy
%       'confu_matrix': confusion matrix

% Parameters recover
is_continuous = config(1);
num_class = config(2);
num_attr = config(3);
D = config(4);
N = size(validLabel, 1);
predict = zeros(N,1);

for i=1:N % for each test point
    scores = p_class;
    for j = 1:num_class
        for d = 1:D
            if is_continuous == 0 %discrete    
                scores(j) = scores(j) * p_condition{testAttributeSet(i,d) + 1}(j,d);
            else                  % continuous
                miu = p_condition{1}(j, d);
                sigma = p_condition{2}(j, d);
                if sigma == 0
                    % should fit with a uniform distribution
                    if(testAttributeSet(i,d) ~= miu)
                        scores(j) = 0;
                    end
                else
                    scores(j) = scores(j) * 1/sqrt(2 * pi * sigma) * exp(-(testAttributeSet(i,d) - miu).^2 / (2 * sigma));
                end
            end
        end
    end
    [MAX, predict(i)] =max(scores);
end
predict = predict - 1;
accuracy = mean(predict == validLabel);
confu_matrix = confusion(predict, validLabel, num_class);