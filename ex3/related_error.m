function error = related_error(x, y)

error = sqrt(mean(sum((x - y).^2, 2)));