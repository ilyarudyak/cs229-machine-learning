function [clusters, centroids] = k_means(X, k)

%%% YOUR CODE HERE

% choose k random different Xi as initial centroids
rng(42);
m = size(X,1);
n = size(X,2);
oldcentroids = zeros(k,n);
centroids = X(randperm(size(X, 1), k), :);

while (norm(oldcentroids - centroids) > 1e-15)
    oldcentroids = centroids;
    % for each point find closest centroid
    for i = 1:m
        [M, clusters(i, 1)] = min(vecnorm(centroids - X(i, :), 2, 2));
    end

    draw_clusters(X, clusters, centroids);
    pause(1);
    
    % recompute centroids
    for i = 1:k
        centroids(i, :) = mean(X(clusters == i, :));
    end
end