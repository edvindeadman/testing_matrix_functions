function [ varargout ] = product_identity(A,AI,f,g,frech_f,frech_g)
%PRODUCT_IDENTITY  Test matrix function algorithms using identity f(A)g(A) = A or I
%   This code computes the relative residual in a matrix function 
%   identity of the form f(A)g(A) = A or f(A)g(A) = I. It then computes an 
%   estimate of the largest relative residual consistent with backward 
%   stability of the algorithms used to compute f(A) and g(A). 
%   Optionally a backward error estimate for the  matrix function 
%   evaluations is also returned.
%
%   Example identities: exp(A)exp(-A) = I; A^{2/3}A^{1/3} = A
%
%   The code can be called in the following ways:
%
%   [res, res_max] = product_identity(A,AI,f,g,frech_f,frech_g)
%   [res, res_max, back_err] = product_identity(A,AI,f,g,frech_f,frech_g)
%
%   where
%       - res is the relative residual ||f(A)g(A)-A||/||A|| or
%         ||f(A)g(A) - I||in the 1-norm.
%       - res_max is the largest 1-norm relative residual consistent with
%         backward stability of the algorithms used to compute f and g.
%       - back_err (optional) is a normwise relative backward error 
%         estimate in the Frobenius norm (section 5 in the reference below).
%       - A is a square matrix.
%       - AI is the righthand side of the identity: this should be A or I.
%       - f and g are function handles for the matrix function algorithms
%         being tested.
%       - frech_f and frech_g are function handles to functions that
%         compute the Frechet derivatives of the matrix functions f and g.
%
%   Reference: E. Deadman and N. J. Higham, Testing Matrix Function
%   Algorithms Using Identities, MIMS EPrint 2014.13

% Compute f(A) and g(A)
gA=feval(g,A);
fA=feval(f,A);

n = size(A,1);

% Compute the normwise relative residual
R = fA*gA-AI;
res = norm(R,1)/norm(AI,1);

% Estimate max||L_f(A,E_1)g(A)+f(A)L_g(A,E_2)||_1 (from eq. (4.19) in ref)
nrm=normest1(@kfun,2);

% Use this to compute res_max
res_max = 2*n*eps*norm(A,1)*nrm/norm(AI,1);

varargout{1} = res;
varargout{2} = res_max;

%If required compute the relative normwise backward error
if nargout == 3
    % Form the n^2 x 2n^2 Kronecker matrix K
    for j=1:2*n
        for i=1:n
            e=zeros(n,1);
            e(i)=1;
            d=zeros(1,2*n);
            d(j)=1;
            Z=e*d;
            E1 = Z(1:n,1:n);
            E2 = Z(1:n,n+1:2*n);
            [f1] = feval(frech_f,A,E1);
            [f2] = feval(frech_g,A,E2);
            u = f1*gA+fA*f2;
            [temp] = vec(u);
            K(:,(j-1)*n+i)=temp;
        end
    end
    
    % Solve the underdeterined linear system by QR factorization
    [r]=vec(R);
    [Q,Z] = qr(K');
    vecE = Q(:,1:n^2)*(Z(1:n^2,1:n^2)'\r);
    % vecE is the minimum 2-norm solution, so backward error
    % estimate will be given in the Frobenius norm
    back_err = norm(vecE,'fro')/norm(A,'fro');
    varargout{3} = back_err;

end

function Z = kfun(flag,x)
   %Function to evaluate matrix products needed by NORMEST1.
   if isequal(flag,'dim')
      Z = 2*n*n;
   elseif isequal(flag,'real')
      Z = isreal(A);
   else
       t = size(x,2);
      if isequal(flag,'notransp')
         % Form Kx and pad out with zeros where necessary.
         for j = 1:t
            [E1] = unvec(x(1:n^2,j),n,n);
            [E2] = unvec(x(n^2 + 1:2*n^2,j),n,n);
            L1 = feval(frech_f,A,E1);
            L2 = feval(frech_g,A,E2);
            tmp = L1*gA+fA*L2;
            [z(1:n^2)]=vec(tmp);
            z(n^2+1:2*n^2) = 0;
            Z(1:2*n^2,j) = z;
         end
      elseif isequal(flag,'transp')
          % Form K'x and padd with zeros where necessary.
          for j = 1:t
            [Y] = unvec(x(:,j),n,n);
            temp2=zeros(n,2*n);
            % These two lines valid if powers series for f and g have real
            % coefficients.
            [temp2(1:n,1:n)]=feval(frech_f,A',Y*gA'); 
            [temp2(1:n,n+1:2*n)]=feval(frech_g,A',fA'*Y);
            [Z(:,j)]=vec(temp2);
          end
      end

   end

end
end

function [v] = vec(X)
n=size(X,1);
m=size(X,2);
v=zeros(m*n,1);
for j=1:m
    v((j-1)*n+1:j*n,1)=X(1:n,j);
end
end

function [X] = unvec(v,rows, cols)
X = zeros(rows,cols);
for j = 1:cols
    X(1:rows,j)=v(1+(j-1)*rows:rows*j);
end
end