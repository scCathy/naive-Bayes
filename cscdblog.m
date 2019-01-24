clc
clear
close all
data=importdata('data.txt');
wholeData=data.data;
%������֤ѡȡѵ�����Ͳ��Լ�        %ѧϰ�˴�
cv=cvpartition(size(wholeData,1),'holdout',0.1);%0.04�����������ݼ�ռ�����ݼ��ı���
trainData=wholeData(training(cv),:); 
testData=wholeData(test(cv),:);
label=data.textdata;
attributeNumber=size(trainData,2);
attributeValueNumber=5;
%�������ǩת��Ϊ����
sampleNumber=size(label,1);
labelData=zeros(sampleNumber,1);
for i=1:sampleNumber
    if label{i,1}=='R'
        labelData(i,1)=1;
    elseif label{i,1}=='B'
        labelData(i,1)=2;
    else 
        labelData(i,1)=3;
    end
end
trainLabel=labelData(training(cv),:);
trainSampleNumber=size(trainLabel,1);
testLabel=labelData(test(cv),:);
%����ÿ������������ĸ���
labelProbability=tabulate(trainLabel);
%P_yi,����P(yi)
P_y1=labelProbability(1,3)/100;
P_y2=labelProbability(2,3)/100;
P_y3=labelProbability(3,3)/100;
%
count_1=zeros(attributeNumber,attributeValueNumber);%count_1(i,j):y=1����£���i������ȡjֵ������ͳ��
count_2=zeros(attributeNumber,attributeValueNumber);%count_1(i,j):y=2����£���i������ȡjֵ������ͳ��
count_3=zeros(attributeNumber,attributeValueNumber);%count_1(i,j):y=3����£���i������ȡjֵ������ͳ��
%ͳ��ÿһ��������ÿ��ȡֵ������
for jj=1:3
    for j=1:trainSampleNumber
        for ii=1:attributeNumber
            for k=1:attributeValueNumber
                if jj==1
                    if trainLabel(j,1)==1&&trainData(j,ii)==k
                        count_1(ii,k)=count_1(ii,k)+1;
                    end
                elseif jj==2
                    if trainLabel(j,1)==2&&trainData(j,ii)==k
                        count_2(ii,k)=count_2(ii,k)+1;
                    end
                else
                    if trainLabel(j,1)==3&&trainData(j,ii)==k
                        count_3(ii,k)=count_3(ii,k)+1;
                    end
                end
            end
        end
    end
end
%�����i������ȡjֵ�ĸ��ʣ�P_a_y1�Ƿ���Ϊy=1ǰ����ȡֵ�������������ơ�
P_a_y1=count_1./labelProbability(1,2);
P_a_y2=count_2./labelProbability(2,2);
P_a_y3=count_3./labelProbability(3,2);
%ʹ�ò��Լ��������ݲ���
labelPredictNumber=zeros(3,1);
predictLabel=zeros(size(testData,1),1);
for kk=1:size(testData,1)
    testDataTemp=testData(kk,:);
    Pxy1=1;
    Pxy2=1;
    Pxy3=1;
    %����P��x|yi��
    for iii=1:attributeNumber
        Pxy1=Pxy1*P_a_y1(iii,testDataTemp(iii));
        Pxy2=Pxy2*P_a_y2(iii,testDataTemp(iii));
        Pxy3=Pxy3*P_a_y3(iii,testDataTemp(iii));
    end
    %����P(x|yi)*P(yi)
    PxyPy1=P_y1*Pxy1;
    PxyPy2=P_y2*Pxy2;
    PxyPy3=P_y3*Pxy3;
    if PxyPy1>PxyPy2&&PxyPy1>PxyPy3
        predictLabel(kk,1)=1;
        disp(['this item belongs to No.',num2str(1),' label or the R label']);
        labelPredictNumber(1,1)=labelPredictNumber(1,1)+1;
    elseif PxyPy2>PxyPy1&&PxyPy2>PxyPy3
        predictLabel(kk,1)=2;
         labelPredictNumber(2,1)=labelPredictNumber(2,1)+1;
        disp(['this item belongs to No.',num2str(2),' label or the B label']);
    elseif   PxyPy3>PxyPy2&&PxyPy3>PxyPy1
        predictLabel(kk,1)=3;
         labelPredictNumber(3,1)=labelPredictNumber(3,1)+1;
        disp(['this item belongs to No.',num2str(3),' label or the L label']);
    end
end
testLabelCount=tabulate(testLabel);
% �����������
disp('the confusion matrix is : ')
C_Bayes=confusionmat(testLabel,predictLabel);
Nb=fitcnb(trainData,trainLabel);
y_nb=Nb.predict(testData);
C_nb=confusionmat(testLabel,y_nb);

% %���ߣ�AiYong_SJTU 
% ��Դ��CSDN 
% ԭ�ģ�https://blog.csdn.net/sjtuai/article/details/75375578 
% ��Ȩ����������Ϊ����ԭ�����£�ת���븽�ϲ������ӣ�