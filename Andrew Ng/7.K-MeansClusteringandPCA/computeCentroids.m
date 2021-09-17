function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
a = zeros(1, n); b = zeros(1, n); c = zeros(1, n);
num = zeros(1, K);
for i=1:m
  if idx(i) == 1
    a(1, :) += X(i, :);
    num(1) += 1;
  end
  if idx(i) == 2
    b(1, :) += X(i, :);
    num(2) += 1;
  end
  if idx(i) == 3
    c(1, :) += X(i, :);
    num(3) += 1;
  end
end

centroids(1, :) = a ./ num(1);
centroids(2, :) = b ./ num(2);
centroids(3, :) = c ./ num(3);

% =============================================================


end

