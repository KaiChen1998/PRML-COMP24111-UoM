function [p_condition, p_class] = NBTrain(AttributeSet, LabelSet, config, use_padding, use_cv)

% Input: 
%       'AttributeSet', 'LabelSet': Training set
%       'config': Parameters used in training time
%       'use_padding': whether or not to use padding to deal with zero
%                   probability
%       'use_cv': whether to use validation to choose best m
%
% Output:
%       'p_condition': Cell, Conditional probability P(xj | ci)
%       'p_class': num_class * 1 vector, P(ci)


% Parameters recover
is_continuous = config(1);  
num_class = config(2);
num_attr = config(3);       
D = config(4);
N = size(LabelSet, 1);

% initialization
p_class = zeros(num_class, 1);
p_condition = {};
validationSet = zeros(floor(N/10), D);
validationLabel = zeros(floor(N/10), 1);

if is_continuous == 0   % discrete
    % Only when dealing with zero probability & using validation we need to bulid validation set
    if use_padding == 1 && use_cv == 1 
        % first shuffle the dataset
        rand_index = randperm(N);
        AttributeSet = AttributeSet(rand_index, :);
        LabelSet = LabelSet(rand_index, :);
        
        % random choose 10% for validation
        validationSet = AttributeSet(1:floor(N/10), :);
        validationLabel = LabelSet(1:floor(N/10), :);
        AttributeSet = AttributeSet(floor(N/10)+1:end, :);
        LabelSet = LabelSet(floor(N/10)+1:end, :);
    end
    
    for i = 1 : num_attr
        p_condition{i} = zeros(num_class, D);
    end
    
else % continuous
    p_condition{1} = zeros(num_class, D); % mean
    p_condition{2} = zeros(num_class, D); % var
end

% Training process
for i = 1:num_class
    % First Calculate p_class
    p_class(i) = mean(LabelSet == i-1); % Label starts with 0
    
    % Bulid classifier in every class space
    index = find(LabelSet == i-1);
    subset = AttributeSet(index, :);
    
    if is_continuous == 0 % discrete
        for j = 1:num_attr % Given certain class, get the conditional probability of every possible value
            p_condition{j}(i,:) = mean(subset == j-1);
        end
    else % continuous
        p_condition{1}(i,:) = mean(subset);
        p_condition{2}(i,:) = var(subset);
    end    
end

if is_continuous == 0 && use_padding == 1
    % Deal with zero probability
    num_class_point = zeros(num_class, 1);
    for i = 1:num_class
        index = find(LabelSet == i-1);
        num_class_point(i) = size(index, 1);
    end
    p_condition = zero_condition(p_condition, p_class, validationSet, validationLabel, num_class_point, config, use_cv);
end    
