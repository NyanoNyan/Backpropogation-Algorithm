clear all

%Training data
Inputs = [1,1;0,1];
Desired = [1,0];

%Input to hidden Layer1
Weights1 = [-0.1,0;0.4,0.2;0,0.05];
%Weights1 = [-0.1,0.05;]; %Weights w1, w6
%Weights1_2 = [ 0.4,0.2]'; %Weights w3, w4

%Hidden to output Layer 2
Weights2 = [0.5,-0.2,0.4]; %Weights w9,w8,w7

%Direct Input Weights
Weights3 = [0.1,-0.2]; %Weights w2, w5
    
LearnRate =1; SumOut1 = 0; SumOut2 = 0; NumLayerNode =5; SigIn = 1;
DataSum = zeros(1,size(Inputs,1));


%Protoptype
% if NumLayerNode>3
%     
%     for i = 4:NumLayerNode
%         Weights1(i,:) = rand(1,2)
%     
%     end
% end


for epoch = 1:100
    
    for k = 1:size(Inputs,1)
     
     x = Inputs(k,:)' ;
     d = Desired(k);;
     
            %Forward Propogation
            
%             Out1st = Weights1'.*x;
%             Out2nd = sum(Weights1_2.*x);
%             Zeros = zeros(3,1);
%             Zeros([1 3]) = Out1st;
%             Zeros(2) = Out2nd;
               
            Net1 = Weights1*x;

            Hidden = 1 ./( 1 + exp( -Net1 ));
            
            %Hidden to out
            Net2 = Weights2'.*Hidden;
            
            %Total Out
            DirectOut = sum(Weights3'.*x);
            SumOut = sum(Net2)+DirectOut;
          
            %Backpropogation
            %Line 89
%            Error = d - SumOut;
%            TSS = sum(sum( Error.^2 ));
            BetaOut = d - SumOut;
            BetaSig = Hidden .* (1-Hidden);
            
            %Local Gradient for Input to Out
            errr = Weights2'.*BetaOut;
            
            BetaHidden = BetaSig .* (errr);
            
            %BetaHidden and Extraction
            q = [1 4];
            p = [1 3];
            
            TestBetaHidden = zeros(4,1);
            Test = BetaHidden(2,1);
            TestBetaHidden(2:3,1) = Test;
            TestBetaHidden(q) = BetaHidden(p);
           
            %WeightCorrection Hidden to Out
            
            deltaW2 = LearnRate * BetaOut * Hidden;
            Weights2 = (Weights2' + deltaW2)';
            
           % This long for Weights 1 training, damm
            
           %Putting BetaHidden in order
            z = zeros(2,2);
            z(1,[1 2]) = [TestBetaHidden(1) TestBetaHidden(2)];
            z(2,[1 2]) = [TestBetaHidden(3) TestBetaHidden(4)];
            
            %Delta w1
            deltaW1 = z'.*x * LearnRate;
           
            WeightTemp = zeros(3,2);
            WeightTemp(1 ,[1 2]) = [deltaW1(1,1) 0];
            WeightTemp(2,[1 2]) = [deltaW1(1,2) deltaW1(2,1)];
            WeightTemp(3 ,[1 2]) = [0 deltaW1(2,2)];

            %Weights Gets changed here for Weights 1
            Weights1 = Weights1 + WeightTemp;
           
            %Delta Rule for Input to Out
            
            Weights3 = (Weights3' + LearnRate*(BetaOut)*x)';
            
            %Display Sum out for different training data
            
            DataSum(1,k) = SumOut;
            
           
    end
            %Mean squared Error
            Error2 = Desired - DataSum;
            TSS = 1/size(Inputs,1)*sum( Error2.^2 );
            %Print Error and SumOut
            fprintf('Epoch %3d:  Error = %f\n',epoch,TSS);
            disp(DataSum);
            gg(epoch)=TSS;
            
            
end

plot(gg);


