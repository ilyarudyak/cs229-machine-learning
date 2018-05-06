function W = ica_st(X)

%%% YOUR CODE HERE
    [n, m] = size(X);
    chunk = 100;
    alpha = 0.0005;
    W = eye(n);

    for iter=1:3
        disp(iter);
        for i = 1:m
            grad = 1 - 2 * sigmoid(W * X(:, i));
            grad = grad * X(:, i)';
            grad += inv(W');
            W += alpha * grad;
        end
    end