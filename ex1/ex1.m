% exercise 1
exe1 = factorial(15)
% exercise 2
A = [16 3 2 13; 5 10 11 8; 9 6 7 12; 4 15 14 1]
B = eye(4,4)
% exercise 3
exe3 = sum(A(:,3))
% exercise 4
exe4 = A(:,1:2:size(A, 2))
% exercise 5
exe5_rand_col = A(:,randperm(4,2))
exe5_rand_row = A(randperm(4,3),:)
% exercise 6
exe6_sum_row = sum(A,2)
exe6_sum_col = sum(A,1)
exe6_sum_diag_1 = sum(diag(A))
exe6_sum_diag_2 = sum(diag(A(1:end,end:-1:1))) %need to rotate the matrix
exe6_sorted = sort(A')'
% exercise 7
exe7_1 = A * B
exe7_2 = A .* B
% exercise 8
exe8 = sum(sum(A > 10))
% exercise 9
exe9 = 1:100;
plot(exe9, log(exe9))
title("y=log(x)")
xlabel("x");
ylabel("y");
% exercise 10
for i=1:4
    for j=1:4
        B(i,j) = 1/A(i,j);
    end
end
exe10_1_for = sum(B)
exe10_2_nofor = sum(1./A)
% exercise 11
% all datapoints in train set and test set range from 0 to 1
% x = rand(1,2);
% A = rand(10,2);
% exe11 = SortDist(x,A)
