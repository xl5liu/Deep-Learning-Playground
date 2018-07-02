%% initialize
clear; close all; clc;

%% load data
load('Ion.trin.mat'); %% include Xtrain and ytrain
load('Ion.test.mat'); %% include Xtest and ytest

%% setup variables 
%%I(input) IW(input weight),A(linear combination fo input),Z(hidden output after non-linearity) HW(hidden weight), yhat(output)
%%m(number of node in hidden layer), d(input dimentsion), n(number of input)

d = 33;
m = 4;
n = 176;
%%add bias to input
one_row = ones(n,1);
I =  [Xtrain; one_row']; %%input, (d+1)*n

%%set y to real output
y = ytrain; %%given output, n*1

%%initialize IW (input weight) with zeros
IW = ones(d+1, m);  %%input weight (d+1)*m
for row = 1:d
    for column = 1:m
        IW(row,column) = randn()*.01;
    end
end

%%initialize HW (hidden weight) with zeros
HW = ones(m+1, 1);
for row = 1:m

    HW(row,1) = randn()*.01;
  
end

%%initialize yhat with zeros;
yhat = zeros(n,1);

%%initialize alpha/learning rate;
alpha = 0.1;

%%initialize lambda/weight decay;
lambda = 0.01;

%%initialize IW_new;
%%IW_new = ones(d+1, m);
%%HW_new = ones(m+1,1);

epoch_num = linspace(1,800,800);
error_vec = zeros(800,1);

%% train NN
epoch_counter = 0;
%TEMP1 = (IW_new - IW).^2;
%TEMP2 = abs(HW_new - HW).^2;
%SUM = sum(TEMP1(:))+sum(TEMP2);
SUM= 100;

while ((SUM > 1e-10) && (epoch_counter < 800)) %%check if the weights do not change further
    epoch_counter = epoch_counter + 1;
    for i = 1:n
        [yhat(i),IW_new,HW_new] = trainNN(I(:,i), y(i), m, IW, HW, alpha, lambda);
        SUM=norm(IW_new-IW)+norm(HW_new - HW)
        IW = IW_new;
        HW = HW_new;

    end
        individual_error_rate = abs(yhat-y);
        error_rate = sum(individual_error_rate)/n;
        error_vec(epoch_counter) = error_rate;
    [I,y] = shuffle(I,y,(d+1),n);

end

%% check error

individual_error_rate = abs(yhat-y);

plot(epoch_num,error_vec)







%% define function for non-linearity
function [sigma_x] = sigma( x )
  if x > 45
      sigma_x = 1;
  elseif x < -45
      sigma_x = 0;
  else    
    sigma_x = 1./(1 + exp(-x));
  end
end

%% define function for derivative of sigma
function [sigma_prime_x] = sigma_prime( x )
    sigma_prime_x = sigma(x)*(1-sigma(x));
end

%% define function for random shuffling columns of a matrix
function [random_X, random_y] = shuffle( matrix, output, row_num, column_num)
    s = RandStream('mt19937ar','Seed',0);
    random_list = randperm(s,column_num);
    random_X = zeros(row_num, column_num);
    random_y = zeros(column_num, 1);
    for i = 1:column_num
        random_X(:,i) = matrix(:,random_list(i));
        random_y(i,1) = output(random_list(i),1);
    end
end

%% define function for run through NN
%%data is a single data point with bias already added
function [prediction, new_A, new_Z, new_a] = runNN( data, hidden_node_num, input_weight, hidden_weight)
    %% calculate hidden layer 
    %%initialize A with 1s
    A = ones(hidden_node_num+1, 1);

    %%initialize z with 1s
    Z = ones(hidden_node_num+1, 1);
     
    hidden_sum_temp = (data'*input_weight)';
    
    
    A(1:hidden_node_num,1) = hidden_sum_temp; %%(m+1)*1 with last term remaining 1    
    Z = sigma(A);
    new_A = A;
    new_Z = Z;
    new_a = Z'* hidden_weight;
    prediction = sigma(new_a); 
    
  
end

%% define function for trainning NN with backpropagation
%%data is a data point with bias added, and real_output is the corresponding real output
%%alpha is the learning rate
%%lambda is the weight decay parameter
function [prediction, new_IW,new_HW] = trainNN(data, real_output, hidden_node_num, input_weight, hidden_weight, alpha, lambda)
    %%run data through NN
    [prediction, new_A, new_Z, new_a] = runNN( data, hidden_node_num, input_weight, hidden_weight);
    
    %%calculate deltas
    delta_last = -2*(real_output - prediction); %% a value
 
    delta_output = sigma_prime(new_a)*delta_last;
    
    delta_hidden = zeros((hidden_node_num + 1), 1); %% initialize with (m+1)*1 zeros
    for i = 1:(hidden_node_num + 1)
       
       delta_hidden(i) = sigma_prime(new_A(i)) * delta_output * hidden_weight(i); %%apply bpp algorithm, no sum becuase delta_output is a value
    end
    
    delta_input = zeros(length(data), 1); %% initialize with (d+1)*1 zeros
 
    for row = 1:length(data)
        row_derivative = sigma_prime(data(row));
       for j = 1:hidden_node_num
            delta_input(row) = delta_input(row) + row_derivative * delta_hidden(j) * input_weight(row,j);
       end
    end
    
    %%calculate gradient
    dEdIW = zeros(length(data), hidden_node_num);
    dEdHW = zeros((hidden_node_num + 1), 1);
    
    for row = 1:length(data)
        for column = 1:hidden_node_num      
            dEdIW(row,column) = delta_hidden(column)*data(row);
        end 
    end
    
    for k = 1:(hidden_node_num + 1)
        dEdHW(k) = delta_output*new_Z(k);
    end
    
    new_IW = input_weight - alpha*(dEdIW + lambda*input_weight);
    new_HW = hidden_weight - alpha*(dEdHW + lambda*hidden_weight);
    
end

