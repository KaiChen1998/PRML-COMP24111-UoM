clear; clc;

% 1: Deal with given datasets
fprintf('----------- Part 1 starts --------\n');
fname = {'av2_c2.mat','av3_c2.mat','av7_c3.mat','avc_c2.mat'};

% record accuracy for 4 given datasets running 10 times
accuracy = zeros(4,10); % without zero probability
accuracy_origin = zeros(4,10); %with zero probability

for j=1:10
    fprintf('Part 1 experiment %d\n',j);
    for i=1:4
        % Pre-process dataset
        [AttributeSet, LabelSet, testAttributeSet, validLabel, config] = data_process(fname{i});

        % original probability with zero
        [p_condition, p_class] = NBTrain(AttributeSet, LabelSet, config, 0, 1); % NB training
        [predict, accuracy_origin(i,j), confu_matrix] = NBTest(testAttributeSet, validLabel, p_condition, p_class, config); % NB test
        % deal with zero probability
        [p_condition, p_class] = NBTrain(AttributeSet, LabelSet, config, 1, 1); % NB training without 0
        [predict, accuracy(i,j), confu_matrix] = NBTest(testAttributeSet, validLabel, p_condition, p_class, config); % NB test

        fprintf('********************************************** \n');
        fprintf('Overall Accuracy on Dataset %s: %f, error with no-padding is %f \n', fname{i}, accuracy(i,j), related_error(accuracy(i,j),accuracy_origin(i,j)));
        fprintf('********************************************** \n');
    end
end
% print result
average = mean(accuracy, 2);
average_origin = mean(accuracy_origin, 2);
for i = 1:4
    fprintf('%s accuracy is %f, origin is %f\n', fname{i}, average(i), average_origin(i));
end
fprintf('related error is %f\n', related_error(average, average_origin));
fprintf('---------- Part 1 finishes ---------\n\n');

% 2: deal with spambase.data
fprintf('---------- Part 2 starts ---------\n');
fname = 'spambase.data';
load(fname);
% randomly shuffle the dataset
rand_index = randperm(size(spambase,1));
spambase = spambase(rand_index,:);
AttributeSet = spambase(:,1:57);
LabelSet = spambase(:,58);
num_each_set = floor(size(AttributeSet,1) / 10); % num_each_set = 460

% bulid config
is_continuous = 1;  num_class = 2;
num_attr = 0;       dim = 57;
config = [is_continuous, num_class, num_attr, dim];

% initialization
accuracy_base = zeros(10,1);
matrix_list = {};
predict_list = {};

for i=1:10
    train = [AttributeSet(1 : (i-1)*num_each_set, :) ; AttributeSet(i*num_each_set+1 : end,:)];
    label_train = [LabelSet(1 : (i-1)*num_each_set) ; LabelSet(i*num_each_set+1 : end)];
    test = AttributeSet((i-1)*num_each_set+1 : i*num_each_set,:);
    label_test = LabelSet((i-1)*num_each_set+1 : i*num_each_set);
    
    [p_condition, p_class] = NBTrain(train, label_train, config, 0, 0);
    [predict_list{i}, accuracy_base(i), matrix_list{i}] = NBTest(test, label_test, p_condition, p_class, config); % NB test
    
    fprintf('********************************************** \n');
    fprintf('Overall Accuracy on Spambase(%d): %f\n', i, accuracy_base(i));
    fprintf('********************************************** \n');
end
fprintf('spam.base mean accuracy is %f\n',mean(accuracy_base));
fprintf('spam.base std is %f\n',std(accuracy_base));
    