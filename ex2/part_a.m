clear; clc;
% load data first
load ORLfacedata;
num_data = 50;  % number of dataset

% Prepare for 50 datasets
% I use an array of matrixs "X_tr" to present the 50 training sets
% Same as Y_tr, X_te and Y_te
% It means X_tr, Y_tr, X_te and Y_te are all (1,50) array, and every
% component of the array is a (N,D) matrix

X = [data(1:10,:) ; data(291:300,:)]; % extract subject 1 and 30
Y = [labels(1:10) ; labels(291:300)]; %(N, 1)
X_tr={};Y_tr={};X_te={};Y_te={};
% repeat 50 times
for i=1:num_data
    [X_tr{i},X_te{i},Y_tr{i},Y_te{i}] = PartitionData(X, Y, 3);
end

% Part 1.a KNN binary classifier
fprintf('Part1 starts\n');
% First compute training and testing accuracy for each k 
k_values = [1,3,5];  % candidates for k
aver_tr = zeros(1, size(k_values,2)); % Given k, average training accuracy on 50 datasets   
std_tr = zeros(1, size(k_values,2)); % Given k, training accuracy std on 50 datasets
aver_te = zeros(1, size(k_values,2)); % Given k, average testing accuracy on 50 datasets  
std_te = zeros(1, size(k_values,2)); % Given k, testing accuracy std on 50 datasets
% Record 2 accuracy of every experiment
accu_tr = zeros(size(k_values,2), num_data); % accu_tr(i,j) means the training accuracy given k = k_values(i) and jth dataset
accu_te = zeros(size(k_values,2), num_data); % accu_tr(i,j) means the training accuracy given k = k_values(i) and jth dataset

is_show = true; % choose whether to show the result
precise_k = zeros(size(k_values,2),2); % prediction precise for each k and each object

for i = 1:size(k_values,2) % for each k
    precise_object = zeros(num_data, 2); %prediction precise for each dataset and each object under certain k
    for j = 1:num_data % for each dataset
        % compute training accuracy
        [predict, accuracy] = KNN_result(k_values(i), X_tr{j}, Y_tr{j}, X_tr{j}, Y_tr{j});
        accu_tr(i,j) = accuracy;
        % compute testing accuracy
        [predict, accuracy] = KNN_result(k_values(i), X_tr{j}, Y_tr{j}, X_te{j}, Y_te{j});
        accu_te(i,j) = accuracy;
        % compute the precise for each object in the test set
        precise_object(j,:) = object_acc(predict, Y_te{j});
        % Show the result on test set
        if is_show == true
            figure(5);
            fprintf('k=%d & this is dataset No.%d, accuracy=%f\n',k_values(i),j,accuracy);
            title('KNN binary classifier test result');
            fprintf('1st object precise is %f & 2nd is %f\n',precise_object(j,1),precise_object(j,2));
            ShowResult(X_te{j}, Y_te{j},predict,7);
            pause;
        end
    end
    precise_k(i,:) = mean(precise_object);
end
% compute the total average prediction precise for each object
precise = mean(precise_k);
fprintf('KNN 1st object precise = %f & 2nd = %f\n',precise(1),precise(2));
% compute mean and std for training and testing accuracy
aver_tr = mean(accu_tr');
std_tr = std(accu_tr');
aver_te = mean(accu_te');
std_te = std(accu_te');

% average accuracy for each dataset
aver_dataset_tr = mean(accu_tr);
aver_dataset_te = mean(accu_te);
index_dataset_tr = find(aver_dataset_tr < 0.85);
index_dataset_te = find(aver_dataset_te < 0.85);
index_dataset = find(aver_dataset_tr < 0.85 & aver_dataset_te < 0.85);

[Max_tr,Max_index_tr] = max(aver_dataset_tr);
fprintf('No.%d dataset has the largest training accuracy %f\n',Max_index_tr, Max_tr);
[Min_tr,Min_index_tr] = min(aver_dataset_tr);
fprintf('No.%d dataset has the smallest training accuracy %f\n',Min_index_tr, Min_tr);
[Max_te,Max_index_te] = max(aver_dataset_te);
fprintf('No.%d dataset has the largest testing accuracy %f\n',Max_index_te, Max_te);
[Min_te,Min_index_te] = min(aver_dataset_te);
fprintf('No.%d dataset has the smallest testing accuracy %f\n',Min_index_te, Min_te);

for i = 1:size(index_dataset,2)
    fprintf('Training and testing accuracy of No. %d are both lower than 0.85\n', index_dataset(i));
end

% plot the training accuracy
figure(1);
errorbar(k_values, aver_tr, std_tr);
title('Part 1.a: KNN Training accuracy of different k values on object 1 and 30');
xlabel('K');
ylabel('Training accuracy');
axis([0,6,0.5,1.1]);
% plot the testing accuracy
figure(2);
errorbar(k_values, aver_te, std_te);
title('Part 1.a: KNN Testing accuracy of different k values on object 1 and 30');
xlabel('K');
ylabel('Testing accuracy');
axis([0,6,0.5,1.1]);
fprintf('Part1 finished\n');

% Part 2.a binary linear classifier
fprintf('Part2 starts\n');
accu_tr_linear = zeros(1,num_data);
accu_te_linear = zeros(1,num_data); 
is_show_linear = true; %choose whether to show the result
precise_object_linear = zeros(num_data, 2); %prediction precise

for i=1:num_data % for each dataset
    % Because it's a binary classification problem
    % I map the label to {0,1} where 0 presents object 1
    % while 1 presents object 30
    labels_tr = Y_tr{i} == 30;
    labels_te = Y_te{i} == 30;
    % add one colomn
    x_tr = [ones(size(X_tr{i},1),1),X_tr{i}];
    x_te = [ones(size(X_te{i},1),1),X_te{i}];
    
    % Normal Equation
    w = pinv((x_tr' * x_tr)) * x_tr' * labels_tr; % (D,1)
    predict = (x_tr * w) > 0.5;
    accu_tr_linear(i) = mean(predict == labels_tr);
    predict = (x_te * w) > 0.5;
    accu_te_linear(i) = mean(predict == labels_te);
    precise_object_linear(i,:) = object_acc(predict, labels_te);
    % show the result
    if is_show_linear == true
        figure(6);
        ShowResult(x_te(:,2:end), labels_te, predict, 7);
        title('Linear Classifier binary classifier test result');
        fprintf('No.%d dataset accuracy is %f\n',i,accu_te_linear(i));
        fprintf('1st object precise=%f & 2nd=%f\n',precise_object_linear(i,1),precise_object_linear(i,2));
        pause;
    end
    fprintf('Part2 %d completed\n',i);
end
precise_linear = mean(precise_object_linear);
fprintf('Linear Classifier average trainging accuracy is %f\n',mean(accu_tr_linear));
fprintf('Linear Classifier average testing accuracy is %f\n',mean(accu_te_linear));
fprintf('LR 1st object precise = %f & 2nd = %f\n',precise_linear(1),precise_linear(2));

% plot the training accuracy
figure(3);
plot(1:num_data,accu_tr_linear);
title('Part 2.a: Linear Classifier Training accuracy on object 1 and 30');
xlabel('Number of Dataset');
ylabel('Training accuracy');
% plot the testing accuracy
figure(4);
plot(1:num_data,accu_te_linear);
title('Part 2.a: Linear Classifier Testing accuracy on object 1 and 30');
xlabel('Number of Dataset');
ylabel('Testing accuracy');

    








	
    