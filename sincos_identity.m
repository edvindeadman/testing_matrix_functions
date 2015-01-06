function [ varargout ] = sincos_identity(A, fsin, fcos, frech_sin, frech_cos)
%SINCOS_IDENTITY  Test matrix function algorithms using identity sin^2(A) + cos^2(A) = I
%   This code computes the relative residual in a matrix function 
%   identity sin^2(A) + cos^2(A) = I. It then computes an 
%   estimate of the largest relative residual consistent with backward 
%   stability of the algorithms used to compute sin(A) and cos(A). 
%   Optionally a backward error estimate for the  matrix function 
%   evaluations is also returned.
%
%   The code can be called in the following ways:
%
%   [res, res_max] = sincos_identity(A,fisn,fcos,frech_sin,frech_cos)
%   [res, res_max, back_err] = sincos_identity(A,fsin,fcos,frech_sin,frech_cos)
%
%   where
%       - res is the residual ||sin^2(A) - cos^2(A)||in the 1-norm.
%       - res_max is the largest 1-norm residual consistent with
%         backward stability of the algorithms used to compute sin and cos.
%       - back_err (optional) is a normwise relative backward error 
%         estimate in the Frobenius norm (see section 5 in the reference
%         below).
%       - A is a square matrix.
%       - fsin and fcos are function handles for the algorithms being
%         tested; fsin returns sin(A) and fcos returns cos(A)
%       - frech_sin and frech_cos are function handles to functions that
%         compute the Frechet derivatives of the matrix functions sin(A)
%         cos(A).
%
%   Reference: E. Deadman and N. J. Higham, Testing Matrix Function
%   Algorithms Using Identities, MIMS EPrint 2014.13


% Compute sin(A) and cos(A)
sinA=feval(fsin,A);
cosA=feval(fcos,A);

n = size(A,1);
I = eye(n);

% Compute the normwise residual
R = sinA^2+cosA^2-I;
res = norm(R,1);

% Estimate ||L_sc(A)||_1 as defined in eq. (4.21) in the reference
nrm=normest1(@kfun,2);

% Use this to compute res_max
res_max = 2*n*eps*norm(A,1)*nrm;

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
            [f1] = feval(frech_sin,A,E1);
            [f2] = feval(frech_cos,A,E2);
            u = sinA*f1 + f1*sinA + cosA*f2+f2*cosA;
            [temp] = vec(u);
            K(:,(j-1)*n+i)=temp;
        end
    end
    E=0;

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
         % Form Kx and pad out with zeros where necessary
         for j = 1:t
        [E1] = unvec(x(1:n^2,j),n,n);
        [E2] = unvec(x(n^2 + 1:2*n^2,j),n,n);
        [L1] = feval(frech_sin,A,E1);
        [L2] = feval(frech_cos,A,E2);
        temp = sinA*L1+L1*sinA + cosA*L2 + L2*cosA;
        [z(1:n^2)]=vec(temp);
        z(n^2+1:2*n^2) = 0;
        Z(1:2*n^2,j) = z;
         end
        %Z=Z';
      elseif isequal(flag,'transp')
          % Form K'x and pad out with zeros where necessary
          for j = 1:t
          [Y] = unvec(x(:,j),n,n);
          temp2=zeros(n,2*n);
          % These two lines valid since powers series for sin and cos have
          % real coefficients.
          [temp2(1:n,1:n)]=feval(frech_sin,A',Y*sinA'+sinA'*Y); 
          [temp2(1:n,n+1:2*n)]=feval(frech_cos,A',Y*cosA'+cosA'*Y);
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