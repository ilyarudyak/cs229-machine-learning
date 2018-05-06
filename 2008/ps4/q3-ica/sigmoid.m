function S = sigmoid(Z)
    S = 1 ./ (1 + exp(-Z));