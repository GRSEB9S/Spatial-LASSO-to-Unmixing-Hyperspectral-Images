
clear
load end4_groundTruth.mat;
load urban_R162.mat;

%%
%%%%%%%%%%%%%%%%%
%    OPTIONS    %
%%%%%%%%%%%%%%%%%

Load_initial_beta = 1;
Segmentation = 1;

%%
%%%%%%%%%%%%%%
%    MAIN    %
%%%%%%%%%%%%%%

% 1. DEFINE PARAMETERS-----------------------------------------------------

fprintf('1. Reading parameters... \n')


M1 = 307;                               % Image size 1
M2 = 307;                               % Image size 2
d = 162;                                % Number of samples wavelengths
m = 4;                                  % Number of endmembers
lambda1 = 1;                            % Sparsity parameters value (note: I am using the same value for all of them)
lambda2 = 100;                          % Spatial regularizer value
k = 1;                                  % Neighborhood size
epsilon = 0.001*M1*M2;                  % Convergence parameter

parameters = struct('M1',M1,...
                    'M2',M2,...
                    'd',d,...
                    'm',m,...
                    'lambda1',lambda1,...
                    'lambda2',lambda2,...
                    'epsilon',epsilon,...
                    'k',k);

% 2. DEFINE DATA ----------------------------------------------------------

fprintf('2. Defining data... \n')

Y = Y(:,1:M1*M2)/1000;                 % Image refectance values [0,1]
X = M;                                 % Dictionary of endmembers
Ycube = reshape(Y,[d,M1,M2]);          % Y data in cube-shape
beta = zeros(m,M1*M2);                 % Proportion of endmembers per pixel

% 3. INITIALIZE PARAMETER beta USING THE STANDARD LASSO (IF NOT LOADED)----

name = strcat('Initial_beta_values_',num2str(parameters.lambda1),'.mat');

if Load_initial_beta == 1
    fprintf('3. Loading initial beta values... \n')
    try
        load(name);
    catch
        error('Initial values for the specified parameters were not found...')
    end
else
    fprintf('3. Computing initial beta values... \n')
    for i=1:(parameters.M1 * parameters.M2)
         if mod(i,5000) == 0
            total = parameters.M1*parameters.M2;
            fprintf('-- 3. Initialization: iteration %d/%d \n',i,total)
         end
         beta(:,i) = l1_ls_nonneg(X,X',size(X,1),size(X,2),Y(:,i),parameters.lambda1,1e-5,1);
    end
    save(name,'beta')
end

[mse,class_e] = mseError(parameters,A,beta);
fprintf('-- 3. LASSO MSE: %.4f \n',mse);

% 4. COMPUTE SPLASSO ------------------------------------------------------

fprintf('4. Computing SPLASSO estimators... \n')

update = Inf;
iteration = 0;
while update > parameters.epsilon
    beta_previous = beta;
    for pixel=1:(M1*M2)
        beta(:,pixel) = Lars_SPLASSO(X,Y,parameters,pixel,beta);
    end
    update = sum(sum(abs(beta_previous-beta)));
    [mse,class_e] = mseError(parameters,A,beta);
    fprintf('-- 4. Iteration %d, mse is %.4f \n',iteration,mse);
    name = strcat('Current_Splasso_solution',...
        '_',num2str(parameters.lambda1),...
        '_',num2str(parameters.lambda2),...
        '_',num2str(parameters.k),'.mat');
    save(name,'beta');
    iteration = iteration + 1;
end

% 5. SEGMENTATION ---------------------------------------------------------

if Segmentation == 1
    fprintf('5. Performing segmentation on the image... \n')
    [d,labels] = max(beta);
    nEnd = length (cood);
    for i = 1 : nEnd
        endmember = i;
        subplot (2,2,i);
        img = zeros(1,parameters.M1*parameters.M2);
        img(labels == endmember) = 1;
        imshow(reshape(img,parameters.M1,parameters.M2));
        title(cood(i));
    end
end

%%
%%%%%%%%%%%%%%%%%%%
%    FUNCTIONS    %
%%%%%%%%%%%%%%%%%%%

function [wrs] = computeWeights(X,Y,parameters,pixel) % Computes weights for one pixel yi
    
    % X: of size dxm, spectrum of the endmember components
    % Y: Y(:,i), spectrum of pixel i
    % k: number of neighbours at each side
    % pixel: pixel being evaluated
    %
    % -------------------------------------- 
    
    k = parameters.k;
    Ycube = reshape(Y,[parameters.d,parameters.M1,parameters.M2]);
    [n,m] = position2pixel(parameters,pixel);
    
    % Spatial Component
    brs = zeros([(2*k+1),(2*k+1)]);
    for i=(n-k):(n+k)
       for j=(m-k):(m+k)
           try
            brs(i-(n-k-1),j-(m-k-1)) = 1/((n-i)^2 + (m-j)^2);
           catch
            brs(i-(n-k-1),j-(m-k-1)) = 0;
           end
       end
    end
    brs(k+1,k+1) = 0;
    brs = brs';
    
    % Spectral Component
    crs = zeros([(2*k+1),(2*k+1)]);
    for i=(n-k):(n+k)
       for j=(m-k):(m+k)
           try
               crs(i-(n-k-1),j-(m-k-1)) = Ycube(:,n,m)'*Ycube(:,i,j)/(norm(Ycube(:,n,m))*norm(Ycube(:,i,j)));
           catch
               crs(i-(n-k-1),j-(m-k-1)) = 0;
           end
       end
    end
    crs(k+1,k+1) = 0;
    crs = crs';
    
    % Final weights
    wrs = brs.*crs;
    wrs = reshape(wrs,[(2*k+1)*(2*k+1),1]);
    
end

function [beta_star,x_star,y_star] = reDefineVectors(X,Y,wrs,parameters,Beta,pixel)
        
    % X: of size dxm, spectrum of the endmember components
    % Y: Y(:,i), spectrum of pixel i
    % wrs: weights capturing the similatiy between the pixel and it neighbors
    % lambda2: spatial regularizer hyperparameter
    % Beta: current estimates of endmember proportions
    % pixel: current pixel from the input image being evaluated
    % k: 2k+1 x 2k+1 is the number of neighbors considered
    %
    % --------------------------------------

    y = Y(:,pixel);
    beta = Beta(:,pixel);
    neighborhood = getNeighbors(parameters,pixel,parameters.k);
    mu = parameters.lambda2.*wrs;
    mu(mu==0)=[];
    
    % Redefine beta_star
    beta_star = sqrt(parameters.lambda2+1).*beta;
    
    % Redefine x_star
    x_star = X;
    for i=1:length(mu)
       x_star = [x_star;  sqrt(mu(i))* eye(parameters.m)];
    end
    x_star = (1/(sqrt(parameters.lambda2+1))) .* x_star;
    
    % Redefine y_star
    y_star = y;
    for i=1:length(mu)
       y_star = [y_star; sqrt(mu(i)).*Beta(:,neighborhood(i))];
    end
end

function [n,m] = position2pixel(parameters,i)
    n = 1+floor(i/parameters.M1);
    m = i - (n-1)*parameters.M1;
    if m == 0
        m = parameters.M1;
        n = n-1;
    end
end

function i = pixel2position(parameters,n,m)
    i = (n-1)*parameters.M1 + m;
    
    if n < 1 || n > parameters.M1 || m < 1 || m > parameters.M2
        i = 0;
    end
end

function neighborhood = getNeighbors(parameters,pixel,k)
    neighborhood = zeros(2*k+1);
    [n,m] = position2pixel(parameters,pixel);
    for i=(n-k):(n+k)
       for j=(m-k):(m+k)
           neighborhood(i-(n-k-1),j-(m-k-1)) = pixel2position(parameters,i,j);
       end
    end
    neighborhood = neighborhood';
    neighborhood(k+1,k+1) = 0;
    neighborhood = reshape(neighborhood,[(2*k+1)*(2*k+1),1]);
    neighborhood(neighborhood==0)=[];
end

function beta = Lars_SPLASSO(X,Y,parameters,pixel,Beta)
    [wrs] = computeWeights(X,Y,parameters,pixel);
    [beta_star,x_star,y_star] = reDefineVectors(X,Y,wrs,parameters,Beta,pixel);
    lambda_star = parameters.lambda1/sqrt(parameters.lambda2+1);
    beta = l1_ls_nonneg(x_star,x_star',size(x_star,1),size(x_star,2),y_star,lambda_star,1e-5,1);
    beta = beta/sqrt(parameters.lambda2+1);
end

function [mse,class_error] = mseError(parameters,A,beta)
    % Classification error
    [M,labels1] = max(beta(:,(1:parameters.M1*parameters.M2)));
    [M,labels2] = max(A(:,(1:parameters.M1*parameters.M2)));
    error = (labels1 ~= labels2);
    class_error = sum(error)/(parameters.M1*parameters.M2);
    
    % mse
    error=(beta(:,(1:parameters.M1*parameters.M2))-A(:,(1:parameters.M1*parameters.M2))).^2;
    mse = sum(sum(error))/(parameters.M1*parameters.M2*parameters.m);    
end

