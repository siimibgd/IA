% Set de date 1
data1 = mvnrnd([1, 1], eye(2), 50);
labels1 = ones(50, 1);

% Set de date 2
data2 = mvnrnd([4, 4], eye(2), 50);
labels2 = -ones(50, 1);

% Concatenarea seturilor de date
data = [data1; data2];
labels = [labels1; labels2];

% Afișarea seturilor de date
figure;
scatter(data(:, 1), data(:, 2), 50, labels, 'filled');
title('Seturi de date cu două roiuri');

% Inițializarea multiplicatorilor Lagrange și bias-ului
alpha = zeros(size(data, 1), 1);
bias = 0;

% Parametri
learning_rate = 0.0001;
num_epochs = 100000;

% Antrenarea SVM
[w, b] = svm_train_linear(data, labels, learning_rate, num_epochs);

% Date de test
test_data = mvnrnd([2, 2], eye(2), 30);

% Predicție
predicted_labels = svm_predict_linear(w, b, test_data);

% Plotare date antrenare
figure;
scatter(data(labels==1, 1), data(labels==1, 2), 'b', 'filled');
hold on;
scatter(data(labels==-1, 1), data(labels==-1, 2), 'r', 'filled');

% Plotare date de test
scatter(test_data(:, 1), test_data(:, 2), 'g', 'filled');

% Plotare linie de decizie pentru SVM cu kernel liniar
x_min = min(data(:, 1)) - 1;
x_max = max(data(:, 1)) + 1;
y_min = min(data(:, 2)) - 1;
y_max = max(data(:, 2)) + 1;

[x, y] = meshgrid(x_min:0.1:x_max, y_min:0.1:y_max);
decision_boundary = w(1) * x + w(2) * y + b;

h = line([x_min, x_max], [-b/w(2) - w(1)/w(2) * x_min, -b/w(2) - w(1)/w(2) * x_max]);
set(h, 'Color', 'k', 'LineWidth', 2);

title('SVM cu kernel liniar - Linia de decizie');
legend('Clasa 1', 'Clasa -1', 'Date de test', 'Linia de decizie');
xlabel('Caracteristica 1');
ylabel('Caracteristica 2');

hold off;