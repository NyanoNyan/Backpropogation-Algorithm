clear all

%Training data
Inputs = [1,1;0,1];
Desired = [1,0];

%Input to hidden Layer1
Weights1 = [-0.1,0;0.4,0.2;0,0.05];

%Hidden to output Layer 2
Weights2 = [0.5,-0.2,0.4]; %Weights w9,w8,w7

Weights3 = [0.1,-0.2];

LearnRate =0.1; NumNodes = 3; activation = 0; Out = 1; NumD = 2;

addNode = input('How many extra nodes: ');

% Addition of Extra Hidden Nodes with random weights
 for i = NumNodes+1:NumNodes+addNode
     
     Weights1(i,:) = rand(1,2);
     Weights2(:,i) = rand(1,1);
     
 end
 
% Addition of Extra Outs
 
 NumNodes = NumNodes+addNode;
 
 addOut = input('How many extra out: ');
 
 if addOut == 0
     addOut =0;
     
 end
 
 % Addition of extra random Desired with extra outs
  for i = size(Desired,2): Out+addOut
     
     for j = 1:size(Desired,2)
         
         Desired(i,j)= randi(0:1)
         
     end
     
 end
 %Addition of Extra Outs with random weights
 for i = Out+1:addOut+Out 
     
     for j = 1:NumNodes
         
         Weights2(i,j) = rand(1);
         
     end
     
 end
 
 Out = Out+addOut;
 
 
 %Zeros variable for storing the datas
 y = zeros(1,NumNodes); t = zeros(1,Out);SumOut = zeros(1,Out);
 
tic
 % NN
for epoch = 1:100
    
for k = 1:size(Inputs,1)
   
    x = Inputs(k,:)' ;
    
    if Out>1
    d = Desired(:,k)';
    
    else
    d = Desired(k);
    
    end
    %Forward Propogation Loop
    
        % Input to Hidden

    for i = 1:NumNodes
            activation = 0;
            
        for j = 1:size(Weights1,2)
            
            activation =+ activation + Weights1(i,j)*x(j);
            
            y(i) = 1.0/(1.0 + exp(-activation));
            y = round(y,4);
            
        end
             % Direct Connections
                Direct = 0;
                
                for a = 1:NumD
                    
                    Direct = Weights3(a)*x(a) + Direct;
                    Direct = round(Direct,4);
                    
                end
    end  
    
    SigOut = 0;
    
    
           % Hidden to Output
            for i = 1: Out
                SigOut = 0;
                for w = 1:size(Weights2,2)
                    
                    SigOut =  round(y(w)*Weights2(i,w) + SigOut,4);
                    t(i) = SigOut;
                    
                    if i == 1 
                        SumOut(i) = Direct+t(i);
                    
                    else
                        SumOut(i) = t(i);
                    end
                    
                end
            
            end
            SumOut = round(SumOut,4);
                
            
% Backpropogation loops
       
% Errors

%Error in Output Nodes
    for i = Out:-1:1
        % For linear activation 
       
        if size(d) ~= size(SumOut)
           
            disp('Size of desired is not set properly')
            
        else
            
            err = round((d - SumOut),4);
            
            
            
        end
        BetaOut(i) =  round(err(i),4);
          
    end
    
    % Error in Hidden Nodes
    for  j = NumNodes:-1:1
        
                bHidden(j) = 0;
                
                for l = 1: Out
                    
                    bHidden(j) = Weights2(j) * BetaOut(l);
                    bHidden(j) = round(y(j) * (1-y(j))* bHidden(j),4);
                    
                    
                end
           
    end
    
    
    %Online Weight Update
    
    % Update for Hidden to Output
        
        for i = 1:Out
            for j = 1:NumNodes
                
                Weights2(i,j) = round(Weights2(i,j) + LearnRate * BetaOut(i) * y(j),4);
                
            end
            
        end
       
     % Update for Input to Hidden
     
        for i = 1:NumNodes
            
            for j = 1:size(Weights1,2)
                
                Weights1(i,j) = round(Weights1(i,j)+ LearnRate * bHidden(i)* x(j),4); 
                
                Weights1(1,2 )=0;
                Weights1(3,1 )=0;
                
            end
            
        end
        
        % Weight update for Direct 
        
        for i = 1:NumD
            
                Weights3(1,i)= round(Weights3(1,i) + LearnRate * (d(1)-SumOut(1,1))*x(i),4);
        end
        
            if k == 1 
               
                SumErr = sum(err);
                SumOutTr1 = SumOut;
            
            elseif k==2
                SumErr = sum(err);
                SumOutTr2 = SumOut;  
                
            end
     
end
%Need to fix errors
MSE = round(1/size(Inputs,1) * sum((d-SumErr).^2),4);


% MSE = 1/size(Inputs,1) * sum(SumErr).^2;
% MSE = round(MSE,4);


fprintf('Epoch %3d:  Error = %0.4f\n',epoch,MSE);

fprintf('Out1 = %g\n',SumOutTr1)
fprintf('Out2 = %g\n',SumOutTr2)
fprintf('\n');

MSE2(epoch) = round(1/size(Inputs,1) * sum((d-SumErr).^2),4);
plot(MSE2)
xlabel('Epoch')
xlim([1 100])
ylabel('MSE')
end
toc
