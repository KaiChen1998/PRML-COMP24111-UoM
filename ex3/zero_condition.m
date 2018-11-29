function result = zero_condition(p_condition, p_class, validationSet, validationLabel, num_class_point, config, use_cv)

% Input: 
%       'p_condition': conditional probability with zero
%       'AttributeSet','LabelSet': train set for cross validation
%       'num_class_point': number of dataponints in each class in training
%                          set
%       'config': useful parameters
%       'use_cv': whether to use validation to choose m
%
% Output:
%       'result' = regularized p_condition

% Parameters recover
num_class = config(2);
num_attr = config(3);
D = config(4);

if use_cv == 0 % use constant m
    m = 1;
    for i = 1:num_attr % for each attr value matrix
        index = find(p_condition{i} == 0);
        [c, d] = address_convert(index, num_class);
        
        if size(index, 1) ~= 0 % zero probability exits
            for j = 1:num_attr
                p_condition{j}(index) = (num_class_point(c) .* p_condition{j}(index) + m / num_attr) ./ (num_class_point(c) + m);
            end
        end
    end
    
else    % do validation to choose the best m
    M = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 15, 20]; % candidate for m

    for i = 1:num_attr % for each attr value matrix
        % find if there is zero probability
        index = find(p_condition{i} == 0);
        [c, d] = address_convert(index, num_class); % index, c and d are of the same size

        if size(index, 1) ~= 0 % zero probability exits
            for k = 1:size(index, 1) % for each zero probability
                best_m = 0; 
                max = 0;
                % compare each candidate m
                for m = 1:size(M, 2) 
                    temp = p_condition;
                    for j = 1:num_attr
                        temp{j}(index(k)) = (num_class_point(c(k)) * temp{j}(index(k)) + M(m) / num_attr) / (num_class_point(c(k)) + M(m));
                    end
                    % validation
                    [~, accuracy, ~] = NBTest(validationSet, validationLabel, temp, p_class, config);
                    if max < accuracy
                        max = accuracy;
                        best_m = M(m);
                    end
                end
                % update p_condition
                for j = 1:num_attr
                    p_condition{j}(index(k)) = (num_class_point(c(k)) * p_condition{j}(index(k)) + best_m / num_attr) / (num_class_point(c(k)) + best_m);
                end
            end
        end
    end
end
result = p_condition;
