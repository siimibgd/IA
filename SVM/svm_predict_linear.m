function predicted_labels = svm_predict_linear(w, b, test_data)
    num_samples = size(test_data, 1);
    predicted_labels = zeros(num_samples, 1);
    
    for i = 1:num_samples
        if dot(w, test_data(i, :)) + b >= 0
            predicted_labels(i) = 1;
        else
            predicted_labels(i) = -1;
        end
    end
end