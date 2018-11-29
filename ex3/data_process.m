function [AttributeSet, LabelSet, testAttributeSet, validLabel, config] = data_process(fname)

% Input:
%       'fname': filename
% Output: 
%       'AttributeSet','LabelSet': Trainset
%       'testAttributeSet','validLabel': Testset
%       'config': Parameters used in training and testing time

% Data set preprocess according to its name
load(fname);
is_continuous = fname(3) == 'c'; % 1 if continuous, or will be 0 
num_class = fname(6) - '0';
num_attr = (fname(3) - '0') * (is_continuous == 0);
dim = size(AttributeSet, 2);
config = [is_continuous, num_class, num_attr, dim];

