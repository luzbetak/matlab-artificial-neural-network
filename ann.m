clc; format shortG;

%----------------- Inputs ---------------------%
%Sample       X       Y      Z    Target ouput
%----------------------------------------------%
Sample   = [  1.0    0.4    0.7    0.65 ];
%Sample  = [ -0.7    0.9    1.2    0.12 ];
%----------------------------------------------%
Learning_Rate = 0.50;

%--- Weights ---%
W1j =  0.20;                     
W1i =  0.10;                

W2j =  0.30;                
W2i = -0.10;                

W3j = -0.10;            
W3i =  0.20;           

Wjk =  0.10;                           
Wik =  0.50;           

%--------------- Neural Network = Input/Output Computation ---------------%
Node_J          = Sample(1)*W1j + Sample(2)*W2j + Sample(3)*W3j;
Output_Node_J   = 1/(1+exp(-Node_J));

Node_I          = Sample(1)*W1i + Sample(2)*W2i + Sample(3)*W3i;
Output_Node_I   = 1/(1+exp(-Node_I));

Input_Node_K    = Output_Node_J*Wjk + Output_Node_I*Wik;
Output_Node_K   = 1/(1+exp(-Input_Node_K));


Expected_Output = Sample(4);
Error_at_Node_K = (Expected_Output-Output_Node_K)*Output_Node_K*(1-Output_Node_K);


Delta_Wjk       = Learning_Rate*Error_at_Node_K*Output_Node_J;
New_Wjk         = Wjk + Delta_Wjk;

Delta_Wik       = Learning_Rate*Error_at_Node_K*Output_Node_I;
New_Wik         = Wik + Delta_Wik;

Error_at_Node_J = Error_at_Node_K * Wjk * Output_Node_J * (1-Output_Node_J);
Error_at_Node_I = Error_at_Node_K * Wik * Output_Node_I * (1-Output_Node_I);


%--------------------------- Node J --------------------------------------%
Delta_W1j       = Learning_Rate * Error_at_Node_J * Sample(1);
New_W1j         = W1j + Delta_W1j;

Delta_W2j       = Learning_Rate * Error_at_Node_J * Sample(2);
New_W2j         = W2j + Delta_W2j;

Delta_W3j       = Learning_Rate * Error_at_Node_J * Sample(3);
New_W3j         = W3j + Delta_W3j;

%--------------------------- Node I --------------------------------------%
Delta_W1i       = Learning_Rate * Error_at_Node_I * Sample(1);
New_W1i         = W1i + Delta_W1i;

Delta_W2i       = Learning_Rate * Error_at_Node_I * Sample(2);
New_W2i         = W2i + Delta_W2i;

Delta_W3i       = Learning_Rate * Error_at_Node_I * Sample(3);
New_W3i         = W3i + Delta_W3i;

%----------------- Weights after current Sample : ------------------------%
Wjk = New_Wjk 
Wik = New_Wik

W1i = New_W1i
W2i = New_W2i 
W3i = New_W3i

W1j = New_W1j
W2j = New_W2j
W3j = New_W3j

%----------------------- End of (ANN)Computation -------------------------%