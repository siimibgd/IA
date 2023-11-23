function [w, b] = svm_train_linear(data, labels, learning_rate, num_epochs)
    [m, n] = size(data);
    
    % Inițializare ponderi și bias
    w = zeros(n, 1);
    b = 0;
    
    % Antrenare SVM cu gradient descent
    for epoch = 1:num_epochs
        for i = 1:m
            if labels(i) * (dot(w, data(i, :)) + b) < 1
                w = w + learning_rate * (labels(i) * data(i, :)' - 2 * w);
                b = b + learning_rate * labels(i);
            else
                w = w - learning_rate * 2 * w;
            end
        end
    end
end
