clc;clear;
data=importdata('data.txt');%非纯数据无法使用load来导入数据
labels=[];
for i=1:length(data.textdata)
    if data.textdata{i}=='L'
        labels(i,1)=1;
    elseif data.textdata{i}=='B'
        labels(i,1)=2;
    elseif data.textdata{i}=='R'
        labels(i,1)=3;
    end
end                     %RBL是否一定要换成数字？将元胞数组转换成数组：cell2mat
numdata=data.data;
len=length(numdata);
traindata=numdata(1:600,:);
testdata=numdata(601:len,:);
lent=length(traindata);
IL=[];IB=[];IR=[];
for i=1:lent
    if labels(i)==1
        IL=[IL i];
    elseif labels(i)==2
        IB=[IB i];
    elseif labels(i)==3
        IR=[IR i];
    end
end                  %根据Y类LBR分类，将其索引值存入IL\IB\IR内，length即为个数
countL=zeros(5,4);countB=zeros(5,4);countR=zeros(5,4);
for i=1:size(traindata,2)
    for n=1:length(IL)
        j=IL(n);
        if traindata(j,i)==1
            countL(1,i)=countL(1,i)+1;
        elseif traindata(j,i)==2
            countL(2,i)=countL(2,i)+1;
        elseif traindata(j,i)==3
            countL(3,i)=countL(3,i)+1;
        elseif traindata(j,i)==4
            countL(4,i)=countL(4,i)+1;
        elseif traindata(j,i)==5
            countL(5,i)=countL(5,i)+1;
        end
    end
end
for i=1:size(traindata,2)
    for n=1:length(IB)
        j=IB(n);
        if traindata(j,i)==1
            countB(1,i)=countB(1,i)+1;
        elseif traindata(j,i)==2
            countB(2,i)=countB(2,i)+1;
        elseif traindata(j,i)==3
            countB(3,i)=countB(3,i)+1;
        elseif traindata(j,i)==4
            countB(4,i)=countB(4,i)+1;
        elseif traindata(j,i)==5
            countB(5,i)=countB(5,i)+1;
        end
    end
end
for i=1:size(traindata,2)
    for n=1:length(IR)
        j=IR(n);
        if traindata(j,i)==1
            countR(1,i)=countR(1,i)+1;
        elseif traindata(j,i)==2
            countR(2,i)=countR(2,i)+1;
        elseif traindata(j,i)==3
            countR(3,i)=countR(3,i)+1;
        elseif traindata(j,i)==4
            countR(4,i)=countR(4,i)+1;
        elseif traindata(j,i)==5
            countR(5,i)=countR(5,i)+1;
        end
    end
end             %计算所有先验概率和条件概率所需数值
%test
error=0;value=zeros(1,25);y=zeros(1,25);testLabel=zeros(1,25);
for i=1:length(testdata)
    py(1)=length(IL)/lent*countL(testdata(i,1),1)/length(IL)*countL(testdata(i,2),2)/length(IL)*countL(testdata(i,3),3)/length(IL)*countL(testdata(i,4),4)/length(IL);
    py(2)=length(IB)/lent*countB(testdata(i,1),1)/length(IB)*countB(testdata(i,2),2)/length(IB)*countB(testdata(i,3),3)/length(IB)*countB(testdata(i,4),4)/length(IB);
    py(3)=length(IR)/lent*countR(testdata(i,1),1)/length(IR)*countR(testdata(i,2),2)/length(IR)*countR(testdata(i,3),3)/length(IR)*countR(testdata(i,4),4)/length(IR);
    [value(i),y(i)]=max(py);%比较获得最大后验概率的类值
    testLabel(i)=labels(i+lent);
    if y(i)~=labels(i+lent)
        i
        error=error+1;
    end
end
correctrate=1-error/length(testdata);   %正确率
C_Bayes=confusionmat(testLabel,y);
trainLabel(1:600)=labels(1:600);
Nb=fitcnb(traindata,trainLabel);
y_nb=Nb.predict(testdata);
C_nb=confusionmat(testLabel,y_nb);