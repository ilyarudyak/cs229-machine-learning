function y = lwlr(X_train, y_train, x, tau)

%%% YOUR CODE HERE

    X = X_train;
    m = size(X,1);
    n = size(X,2);
    lambda = 1e-4;
    theta = zeros(n,1); % n x 1 - the same as x(i)

    % compute weights, size = m x 1
    w = exp(-(vecnorm(X - x', 2, 2) .^ 2)  / (2 * tau ^ 2));

    % perform Newton’s method 
    grad = ones(n,1); % n x 1 - the same as theta
    while (norm(grad) > 1e-6)
        h = 1 ./ (1 + exp(-X * theta)); % m x 1
        z = w .* (y_train - h); % m x 1
        D = diag(-w .* h .* (1 - h)); % m x m
        
        grad = X' * z - lambda * theta;
        H = X' * D * X - lambda * eye(n); % n x n
        theta = theta - inv(H) * grad;
    end

    % return predicted y
    h = 1 ./ (1 + exp(-x' * theta));
    y = double(h > .5);

