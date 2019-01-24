clc;clear;
data=importdata('data.txt');%�Ǵ������޷�ʹ��load����������
labels=cell2mat(data.textdata);IL=[];IB=[];IR=[];
numdata=data.data;
len=length(numdata);
cv=cvpartition(len,'holdout',0.04);%0.04�����������ݼ�ռ�����ݼ��ı���
traindata=numdata(training(cv),:);
testdata=numdata(test(cv),:);
trainlabel=labels(training(cv),:);
testlabel=labels(test(cv),:);
lent=length(traindata);
for i=1:lent
    if trainlabel(i)=='L'
        IL=[IL i];
    elseif trainlabel(i)=='B'
        IB=[IB i];
    elseif trainlabel(i)=='R'
        IR=[IR i];
    end
end                  %����Y��LBR���࣬��������ֵ����IL\IB\IR�ڣ�length��Ϊ����
countL=zeros(5,4);countB=zeros(5,4);countR=zeros(5,4);
IY={IL;IB;IR};count={countL;countB;countR};
for m=1:3
    for i=1:size(traindata,2)
        for n=1:length(IY{m,1})
            j=IY{m,1}(n);
        if traindata(j,i)==1
            count{m,1}(1,i)=count{m,1}(1,i)+1;
        elseif traindata(j,i)==2
            count{m,1}(2,i)=count{m,1}(2,i)+1;
        elseif traindata(j,i)==3
            count{m,1}(3,i)=count{m,1}(3,i)+1;
        elseif traindata(j,i)==4
            count{m,1}(4,i)=count{m,1}(4,i)+1;
        elseif traindata(j,i)==5
            count{m,1}(5,i)=count{m,1}(5,i)+1;
        end
        end
    end
end
countL=count{1,1};countB=count{2,1};countR=count{3,1};
%��������������ʺ���������������ֵ
%test
testlen=length(testdata);
error=0;value=zeros(1,testlen);y=zeros(1,testlen);class=['L','B','R'];
for i=1:testlen
    L=length(IL)/lent*countL(testdata(i,1),1)/length(IL)*countL(testdata(i,2),2)/length(IL)*countL(testdata(i,3),3)/length(IL)*countL(testdata(i,4),4)/length(IL);
    B=length(IB)/lent*countB(testdata(i,1),1)/length(IB)*countB(testdata(i,2),2)/length(IB)*countB(testdata(i,3),3)/length(IB)*countB(testdata(i,4),4)/length(IB);
    R=length(IR)/lent*countR(testdata(i,1),1)/length(IR)*countR(testdata(i,2),2)/length(IR)*countR(testdata(i,3),3)/length(IR)*countR(testdata(i,4),4)/length(IR);
    [value(i),y(i)]=max([L,B,R]);%�Ƚϻ����������ʵ���ֵ
    if class(y(i))~=testlabel(i)
         disp('Ԥ���������к�'),i
         error=error+1;
     end
end
correctrate=1-error/length(testdata);   %��ȷ��
%C_Bayes=confusionmat(testlabel,y);
%matlab�Դ����ر�Ҷ˹������֤,����Ϊ������ʽ
% Nb=fitcnb(traindata,trainlabel);
% y_nb=Nb.predict(testdata);
% C_nb=confusionmat(testlabel,y_nb);