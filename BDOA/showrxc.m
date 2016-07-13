function showrxc(W,r,c)

if nargin < 2; r = 4; end
if nargin < 3; c = 4; end

m = sqrt(size(W,1));
Panel = col2im(W(:,1:r*c),[m m],[r*m c*m],'distinct');
imshow(Panel,[]);

