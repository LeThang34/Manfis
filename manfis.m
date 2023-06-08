% Load data
thpt_data = readtable('thpt.csv');

% Define input and output variables
X = table2array(thpt_data(:, {'Toan', 'Van', 'Ly', 'Hoa', 'Sinh', 'Su', 'Dia', 'GDCD', 'Anh'}));
Y = table2array(thpt_data(:, {'A00', 'A01', 'B00', 'B08', 'C00', 'D00'}));

% Define the number of rules and the type and parameters of the membership function
num_rules = 9;
mf_type = 'bell';
mf_params = [1.5, 0.3, 0.5]; % You can adjust these parameters

% Create the ANFIS model with default parameters
fis = genfis(X, Y(:, i), 'sugeno', num_rules); 

% Modify the membership function parameters
for j=1:size(X,2)
    fis.input(j).mf(1).params = mf_params;
end

% Set the options for the PSO optimization algorithm
options = psoptimset('Display', 'none', 'MaxIter', 1000, 'MaxFunEvals', 10000);

% Define the objective function for the PSO algorithm
objective_function = @(w)compute_rmse(w, X, Y, fis);

% Run the PSO algorithm to optimize the output weights
num_outputs = size(Y, 2);
output_weights = zeros(size(X,2), num_outputs);
for i = 1:num_outputs
    [output_weights(:,i), ~] = pso(objective_function, size(X,2), [], [], [], [], [], [], options);
end

% Calculate the predicted output for the entire dataset
Y_pred = evalfis(X, fis);
Y_pred = [Y_pred(:,1), Y_pred(:,2), Y_pred(:,3), Y_pred(:,4), Y_pred(:,5), Y_pred(:,6)];

% Display the results
disp(Y_pred);
