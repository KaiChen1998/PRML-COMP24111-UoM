function result = SortDist(x, A)
dist = A - x;
dist = sqrt(sum(dist.^2, 2));
[a, b] = sort(dist);
result = A(b(1:3),:);