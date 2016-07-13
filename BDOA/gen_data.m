clear;

%testcase = 1; % numbers
%testcase = 2; % planes
%testcase = 2; % random
testcase = input('testcase (1 to 3) = ');
if isempty(testcase); testcase = 1; end

% get Ws
switch testcase
    case 1;
        load Utilities/Digits;
        gsize = [4 4];
        p = 8; 
    case 2;
        load Utilities/PlaneParts;
        gsize = [2 3 3];
        p = 10; 
    otherwise;
        m = 10; n = m; K = 6;
        Ws = rand(m*n,K) > .6;
        Ws = Ws*sparse(1:K,1:K,1./max(Ws),K,K);
        gsize = [2 4];
        p = 6;
end

% number of data vectors = p*q
q = p;
d = m*n; 
N = p*q;
K = sum(gsize);

Hs = zeros(N,K);
gs = [0 cumsum(gsize)];

%s = RandStream('mt19937ar','seed',5489);
%RandStream.setDefaultStream(s);

% get Hs
rho = 0.3;
for k = 1:numel(gsize)
    Jk = gs(k)+1:gs(k+1);
    [~,Ik] = max(rand(N,gsize(k)),[],2);
    Ck = rho + (1-rho)*rand(N,1);
    Hs(:,Jk) = sparse((1:N)',Ik,Ck,N,gsize(k));
end

% generate the data matrix
Ao = Ws*Hs';
fprintf('Data generated with N = %i\n',N)
clearvars -except Ws Hs m n p q gsize Ao

%warning off; showrxc(Ao,p,q);