
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Examples demonstrating the use of the mfiles composition_identity %%%%
%%%% product identity and sincos_identity got testing matrix functions %%%%
%%%% algorithms using identities.                                      %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = examples()
% EXAMPLES   Demonstrate use of the 'Testing Matrix Functions' mfiles

% Create a small random matrix and square it to ensure we'll be able to
% compute matrix logarithms and roots when we want to.
A = rand(4)^2

% Test the identity sin^2(A) + cos^2(A) = I using Matlab's funm and the
% complex step approximation for the Frechet derivatives
disp('Testing sin^2(A) + cos^2(A) = I:')
[res,res_max,back_err] = sincos_identity(A,@sinm,@cosm,@frech_sin,@frech_cos)

% We should have obtained res < res_max for backward stability and the
% backward error, back_err, should be 'close' to 1e-16.

% The next tests require some routines for computing Frechet derivatives
% and condition numbers of matrix powers, logarithms and exponentials.
% We use codes available from the following links:

% The Matrix Function Toolbox: http://www.ma.man.ac.uk/~higham/mftoolbox/
% Fractional matrix powers: http://www.mathworks.com/matlabcentral/fileexchange/41621-fractional-matrix-powers-with-frechet-derivatives-and-condition-number-estimate
% For a list of available Matlab matrix funciton codes see: http://eprints.ma.man.ac.uk/2102/01/covered/MIMS_ep2014_8.pdf

% Test the identity exp(log(A)) = A using Matlab's expm and logm and
% Frechet derivative codes from the Matrix Function Toolbox:
disp('Testing exp(log(A)) = A:')
[res,res_max, back_err] = composition_identity(A,@expm,@logm,@expm_cond,@expm_frechet_pade)

% Test the identity exp(A)exp(-A) = I using Matlab's expm and expm_frechet_pade
% from the Matrix Function Toolbox with the functions defined below for
% computing exp(-A) and its derivative:
disp('Testing exp(A)exp(-A) = I:')
[res, res_max, back_err] = product_identity(A,eye(4),@expm,@expmm,@expm_frechet_pade,@expmm_frechet_pade)

% Test the identity A^{1/3}A^{2/3} = A using Matlab's the powerm_pade
% routines (see the functions below):
disp('Testing A^{1/3}A^{2/3} = A:')
[res, res_max, back_err] = product_identity(A,A,@A13,@A23,@frech_A13,@frech_A23)


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% Functions used in the examples above %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function sinA = sinm(A)
% Compute sin(A)
sinA = funm(A,@sin);
end

function cosA = cosm(A)
% Compute cos(A)
cosA = funm(A,@cos);
end

function L = frech_sin(A,E)
% Compute the Frechet derivative of sin(A) using complex step approximation
% see Functions of Matrices, Higham 2008, section 3.4
if norm(E,1) == 0
    n=size(E,1);
    L=zeros(n,n);
else
    lt = ((eps*norm(funm(A,@sin),1)/2)^(1/3))/norm(E,1);
    L = imag(funm(A+sqrt(-1)*lt*E,@sin))/lt;
end
end

function L = frech_cos(A,E)
% Compute the Frechet derivative of cos(A) using complex step approximation
% see Functions of Matrices, Higham 2008, section 3.4
if norm(E,1) ==0
    n=size(E,1);
    L=zeros(n,n);
else
    lt = ((eps*norm(funm(A,@cos),1)/2)^(1/3))/norm(E,1);
    L = imag(funm(A+sqrt(-1)*lt*E,@cos))/lt;
end
end

function F = A13(A)
% Compute A^{1/3} using powerm_pade_fre
F = powerm_pade_fre(A,1/3);
end

function F = A23(A)
% Compute A^{2/3} using powerm_pade_fre
F = powerm_pade_fre(A,2/3);
end

function L = frech_A13(A,E)
% Compute the Frechet derivative of A^{1/3} using powerm_pade_fre
[F,L]=powerm_pade_fre(A,1/3,E);
end

function L = frech_A23(A,E)
% Compute the Frechet derivative of A^{2/3} using powerm_pade_fre
[F,L]=powerm_pade_fre(A,2/3,E);
end

function F =expmm(A)
% Compute exp(-A) using Matlab's expm
F = expm(-A);
end

function F = expmm_frechet_pade(A,E)
% Compute the Frechet derivative of exp(-A) using expm_frechet_pade
F = expm_frechet_pade(-A,E);
end