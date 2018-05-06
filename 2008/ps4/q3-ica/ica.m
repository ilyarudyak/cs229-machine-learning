function W = ica(X)

%%% YOUR CODE HERE
    [n, m] = size(X);
    chunk = 100;
    alpha = 0.0005;
    W = eye(n);

    for iter=1:10
        disp([num2str(iter)]);
        X = X(:, randperm(m));
        for i = 1:floor(m/chunk)
            Xc = X(:, (i-1)*chunk+1:i*chunk);
            grad = 1 - 2 * sigmoid(W * Xc);
            grad = grad * Xc';
            grad += inv(W') * chunk;
            W += alpha * grad;
        end
    end