clear; clc;
% first load data
load ORLfacedata;
num_data = 50;
C = 40; % num of classes
X = data(1:400,:);
Y = labels(1:400);
X_tr={};Y_tr={};X_te={};Y_te={};

for i=1:num_data
    [X_tr{i},X_te{i},Y_tr{i},Y_te{i}] = PartitionData(X, Y, 5);
end

% Part 1.b
fprintf('Part1 starts\n');
rand_index = randperm(num_data,1); % random pick one dataset for CV
k_value = [1,3,5,7,9,11,13,15,17,19,23,29]; % candidate for k
acc_vector = zeros(1,size(k_value,2)); % accuracy for each k in dataset(rand_index)

% leave-one-out crossvalidation
for i=1:size(k_value,2) % for each k
    Sum = 0; % count for correct predict
    for j=1:200 % for each datapoint in training set (as validation set)     
        % predict here is a scalar
        predict = knearest(k_value(i), X_tr{rand_index}(j,:), [X_tr{rand_index}(1:(j-1),:) ; X_tr{rand_index}((j+1):end,:)], [Y_tr{rand_index}(1:(j-1)) ; Y_tr{rand_index}((j+1):end)]);
        Sum = Sum + (predict == Y_tr{rand_index}(j));
        fprintf('k=%d & this is CV No.%d\n',k_value(i),j);
    end
    acc_vector(i) = Sum / 200;
end
[Max, index] = max(acc_vector);
k = k_value(index);
fprintf('Best k is %d\n',k);
%compute test accuracy in this dataset(rand_index)
[predict, accuracy] = KNN_result(k, X_tr{rand_index}, Y_tr{rand_index}, X_te{rand_index}, Y_te{rand_index});
fprintf('No.%d dataset test accuracy is %f\n',rand_index, accuracy);

%compute 50 datasets' test accuracy and std
accu_te = zeros(1, num_data); 
precise_object = zeros(num_data, C);
for j = 1:num_data
    [predict, accuracy] = KNN_result(k, X_tr{j}, Y_tr{j}, X_te{j}, Y_te{j});
    accu_te(j) = accuracy;
    precise_object(j,:) = object_acc(predict, Y_te{j}); %prediction precise
    fprintf('No.%d dataset completed, accuracy=%f\n', j, accuracy);
end
precise = mean(precise_object);
aver_te = mean(accu_te);
std_te = std(accu_te);
fprintf('KNN Average accuracy is %f\n',aver_te);
fprintf('KNN Standard deviation is %f\n',std_te);
% print precise of the 10 most difficult for KNN
[order, index] = sort(precise);
for i=1:10
    fprintf('No.%d most difficult for KNN is %d with precise %f\n',i,index(i),precise(index(i)));
end

% plot the test accuracy
figure(1);
plot(1:50,accu_te,'b-');
hold on;
fprintf('Part1 finishes\n');

% Part 2.b
fprintf('Part2 starts\n');
accu_te_linear = zeros(1,num_data); 
precise_object_linear = zeros(num_data, C);
for i=1:num_data
    x_tr = [ones(size(X_tr{i},1),1),X_tr{i}]; %add one colomn
    x_te = [ones(size(X_te{i},1),1),X_te{i}]; %(N,D)
    % spread the label vector to a one hot matrix
    N = size(x_tr,1); % size of training set
    label_tr = zeros(N, C); %(N,C)
    for j=1:200 
        label_tr(j,Y_tr{i}(j)) = 1;
    end
    % Normal Equation
    w = pinv((x_tr' * x_tr)) * x_tr' * label_tr; % (D,C)
    [Max,predict] = max(x_te * w,[],2); % (N,D)*(D,C)
    accu_te_linear(i) = mean(predict == Y_te{i});
    precise_object_linear(i,:) = object_acc(predict, Y_te{i});
    fprintf('%d completed,accuracy=%f\n',i,accu_te_linear(i));
end
precise_linear = mean(precise_object_linear);
fprintf('Linear Classifier Average accuracy is %f\n',mean(accu_te_linear));
fprintf('Linear Classifier Standard deviation is %f\n',std(accu_te_linear));

% print precise of the 10 most difficult for LR
[order, index] = sort(precise_linear);
for i=1:10
    fprintf('No.%d most difficult for LR is %d with precise %f\n',i,index(i),precise_linear(index(i)));
end
% plot the test accuracy
plot(1:50,accu_te_linear,'r-');
legend('KNN','Linear');
title('Part 1.b & 2.b: KNN & Linear Classifier test accuracy');
xlabel('No. of dataset');
ylabel('Test accuracy');

figure(2);
plot(1:40,precise,'b-');
hold on;
plot(1:40,precise_linear,'r-');
legend('KNN','Linear');
title('Part 1.b & 2.b: KNN & Linear precise for each object');
xlabel('No. of object');
ylabel('Precise');









