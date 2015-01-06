function [ varargout ] = composition_identity(A,f,g,cond_f,frech_f)
%COMPOSITION_IDENTITY  Test matrix function algorithms using an identity f(g(A)) = A
%   This code computes the relative residual in the matrix function 
%   identity of the form f(g(A)) = A. It then computes an estimate of the
%   largest relative residual consistent with backward stability of the
%   algorithms used to compute g(A) and f(g(A)). Optionally a backward
%   error estimate for the  matrix function evaluations is also returned.
%
%   Example identities: exp(log(A)) = A; (A^{1/2})^2 = A
%
%   The code can be called in the following ways:
%
%   [res, res_max] = composition_identity(A,f,g,cond_f,frech_f)
%   [res, res_max, back_err] = composition_identity(A,f,g,cond_f,frech_f)
%
%   where
%       - res is the relative residual ||f(g(A))-A||/||A|| in the 1-norm.
%       - res_max is the largest 1-norm relative residual consistent with
%         backward stability of the algorithms used to compute f and g.
%       - back_err (optional) is a normwise relative backward error 
%         estimate in the Frobenius norm (see the first equation of section
%         5 in the reference below).
%       - A is a square matrix.
%       - f and g are function handles for the matrix function algorithms
%         being tested.
%       - cond_f is a function handle to a function that computes or
%         estimates the 1-norm condition number of the matrix function f.
%       - frech_f is a function handle to a function that computes the
%         Frechet derivative of the matrix function f.
%
%   Reference: E. Deadman and N. J. Higham, Testing Matrix Function
%   Algorithms Using Identities, MIMS EPrint 2014.13

% Compute f(g(A))
gA=feval(g,A);
fgA=feval(f,gA);

% Compute condition number of f at g(A)
[kappa]=feval(cond_f,gA);

% Estimate res_max; maximum residual consistent with backward stability
n=size(A,1);
res_max = n*eps*(1+kappa)/2;

% Compute the normwise relative residual
R = fgA-A;
res = norm(R,1)/norm(A,1);

varargout{1} = res;
varargout{2} = res_max;

% If required compute the relative normwise backward error
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
            [fr] = feval(frech_f,gA,E2);
            Y = E1+fr;
            [temp] = vec(Y);
            K(:,(j-1)*n+i)=temp;
        end
    end
    E=0;

    % Solve the underdeterined linear system by QR factorization
    [r] = vec(R);
    % Some scaling is required - see eq. (5.23) in the reference
    normA = norm(A,2);
    normgA = norm(gA,2);
    Kscal(1:n^2,1:n^2) = K(1:n^2,1:n^2)*normA;
    Kscal(1:n^2,n^2+1:2*n^2) = K(1:n^2,n^2+1:2*n^2)*normgA;
    [Q,Z] = qr(Kscal');
    vecE = Q(:,1:n^2)*(Z(1:n^2,1:n^2)'\r);
    % vecE is the minimum 2-norm solution, so backward error
    % estimate will be given in the Frobenius norm
    back_err = norm(vecE,'fro');
    varargout{3} = back_err;

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