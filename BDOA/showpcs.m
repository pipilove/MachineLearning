function showpcs(W)

% display basis W in a 2-row image array

[m2,n] = size(W); m = sqrt(m2);
edge = 2; A = [W; ones(edge*m,n)]; L = n/2;
Panel = col2im(A,[m m+edge],[2*m L*(m+edge)],'distinct');
Panel = [Panel(1:m,:); ones(2,L*(m+edge)); Panel(m+1:end,:)];

%h = figure; 
imshow(Panel,[]); 