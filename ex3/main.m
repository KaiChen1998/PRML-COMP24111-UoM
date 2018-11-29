clear; clc;

fname = input('Enter a filename to load data for training/testing: ','s');
if strcmp(fname, 'spambase.data') == 0 % Deal with given datasets
    % Pre-process dataset
    [AttributeSet, LabelSet, testAttributeSet, validLabel, config] = data_process(fname);
    accuracy = zeros(10, 1);
    accuracy_origin = zeros(10, 1);
    
    for i = 1:10
        % train with zero probability
        [p_condition, p_class] = NBTrain(AttributeSet, LabelSet, config, 0, 1); % NB training
        [predict, accuracy_origin(i), confu_matrix] = NBTest(testAttributeSet, validLabel, p_condition, p_class, config); % NB test

        % train without zero probability
        [p_condition, p_class] = NBTrain(AttributeSet, LabelSet, config, 1, 1); % NB training
        [predict, accuracy(i), confu_matrix] = NBTest(testAttributeSet, validLabel, p_condition, p_class, config); % NB test
    end
        
    fprintf('********************************************** \n');
    fprintf('Overall Accuracy on Dataset %s: %f, error with no-padding is %f \n', fname, mean(accuracy), related_error(mean(accuracy), mean(accuracy_origin)));
    fprintf('********************************************** \n');
    
else  % deal with spambase.data
    load(fname);
    % randomly shuffle the dataset
    rand_index = randperm(size(spambase,1));
    spambase = spambase(rand_index,:);
    
    AttributeSet = spambase(:,1:57);
    LabelSet = spambase(:,58);
    num_each_set = floor(size(AttributeSet,1) / 10); % num_each_set = 460
    
    is_continuous = 1;  num_class = 2;
    num_attr = 0;       dim = 57;
    config = [is_continuous, num_class, num_attr, dim];
    accuracy_base = zeros(10,1);
    matrix_list = {};
    predict_list = {};
    
    for i=1:10
        train = [AttributeSet(1 : (i-1)*num_each_set, :) ; AttributeSet(i*num_each_set+1 : end,:)];
        label_train = [LabelSet(1 : (i-1)*num_each_set) ; LabelSet(i*num_each_set+1 : end)];
        test = AttributeSet((i-1)*num_each_set+1 : i*num_each_set,:);
        label_test = LabelSet((i-1)*num_each_set+1 : i*num_each_set);
        
        [p_condition, p_class] = NBTrain(train, label_train, config, 0, 0); % NB training
        [predict_list{i}, accuracy_base(i), matrix_list{i}] = NBTest(test, label_test, p_condition, p_class, config); % NB test
        
        fprintf('********************************************** \n');
        fprintf('Overall Accuracy on Spambase(%d): %f\n', i, accuracy_base(i));
        fprintf('********************************************** \n');
    end
    fprintf('spam.base mean accuracy is %f\n',mean(accuracy_base));
    fprintf('spam.base std is %f\n',std(accuracy_base));
end