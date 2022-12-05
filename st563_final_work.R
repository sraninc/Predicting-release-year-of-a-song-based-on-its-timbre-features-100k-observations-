library(ggplot2)
library(dplyr)
library(tidyverse)
library(GGally)
library(ggpubr)
library(e1071)
library(leaps)
library(groupdata2)
set.seed(1)

## Load in Original Data
Train<-read.table('YearPredictTrain.txt',header = TRUE, sep=',')
Test<-read.table('YearPredictTest.txt',header=TRUE,sep=',')

## Combine Test and Training Data for Subset Selection
Full<-rbind(Train,Test)

## Remove Quantitative Response
Full<-Full[,-91]

## Create dataframe for full dataset
Full_pred<-Full[,1:90]
Full_resp<-Full[,91]
Full_data<-data.frame(x=Full_pred, y=as.factor(Full_resp))
table(Full$Class)

## downsample full data
Downsamp_Full<-downsample(Full_data, cat_col = "y", id_method = "n_ids")
table(Downsamp_Full$y)

## PCA
XTrain_Class<-Train[,1:90]
YTrain_Class<-Train[,92]
data<-data.frame(x=XTrain_Class,y=as.factor(YTrain_Class))
pca <- prcomp(XTrain_Class, scale. = TRUE)
pca_score <- pca$x
summary(pca)
##PCA results indicate that 90% of variation is explained by fisrt 55PCs, so not very helpful tool
##First 2 PCs only explain 19%

##Forward stepwise selection Full Data
forward<-regsubsets(y~., data=Full_data, nvmax=90, method="forward")
mod_summary<-summary(forward)
mod_summary
metrics <- data.frame(aic = mod_summary$cp,
                      bic = mod_summary$bic,
                      adjR2 = mod_summary$adjr2)
metrics
x=seq(1,90,1)
aic<-plot(x,metrics$aic)
bic<-plot(x,metrics$bic)
adjr<-plot(x,metrics$adjR2)
combo<-ggarrange(aic, bic, adjr,
          nrow=1, ncol=3)
print(combo)
round(coef(forward,53),4)
##Forward Stepwise Selection indicates the best model contain only 53 predicotrs instead of 90

##Building training set with only best 53 predictors
best_forward_preds<-c('timbreAvg1',
                      'timbreAvg2','timbreAvg3','timbreAvg5',
                      'timbreAvg6', 'timbreAvg8','timbreAvg9',
                      'timbreAvg10','timbreAvg11','timbreCov1',
                      'timbreCov2', 'timbreCov3','timbreCov4',
                      'timbreCov5', 'timbreCov6','timbreCov7',
                      'timbreCov8','timbreCov9','timbreCov11',
                      'timbreCov12', 'timbreCov13','timbreCov15',
                      'timbreCov20','timbreCov21','timbreCov24',
                      'timbreCov26', 'timbreCov27','timbreCov28',
                      'timbreCov29', 'timbreCov34','timbreCov35',
                      'timbreCov36','timbreCov38','timbreCov41',
                      'timbreCov45', 'timbreCov46','timbreCov47',
                      'timbreCov49','timbreCov51','timbreCov52',
                      'timbreCov53', 'timbreCov57','timbreCov58',
                      'timbreCov59', 'timbreCov62','timbreCov63',
                      'timbreCov64','timbreCov66','timbreCov71',
                      'timbreCov73', 'timbreCov75','timbreCov76','timbreCov77')
XTrain_Forward<-XTrain_Class[,best_forward_preds]
YTrain_Class<-Train[,92]
data_for<-data.frame(x=XTrain_Forward,y=as.factor(YTrain_Class))

##Reattempting SVM with best 53 pred model
ds_data_for<-downsample(data_for,cat_col="y",id_method="n_ids")
tuning<-tune(svm, y~., data= ds_data_for, kernel='radial', ranges=list(cost=c(0.1,1,10,100,1000),gamma=c(0.5,1,2,3,4)))
out<-svm(y~.,data=ds_data_for,kernel="radial",cost=10, gamma=1)
out
plot(out,ds_data_for)
pred_svm <- predict(object = out,
                      newdata = test_data,
                      type = "response")
err <- klaR::errormatrix(true = test_data$y,
                         predicted = pred_svm,
                         relative = TRUE)
round(err, 3)

tr <- trainControl(method = "repeatedcv",
                   number = 5, repeats = 10)
tune_grid <- expand.grid(cost = exp(seq(-5,3,len=30)))
sv_caret <- train(as.factor(y) ~ .,data = ds_data_for,
                  method = "svmLinear2",tuneGrid = tune_grid,
                  trControl = tr)

#QDA
library(MASS)
library(caret)

Test_pred<-Test[,1:90]
Test_rep<-Test[,92]
test_data<-data.frame(x=Test_pred,y=as.factor(Test_rep))

QDA_ds<-qda(y~., data=ds_data_for)
pred_qda_ds <- predict(QDA_ds, newdata =test_data)
pred_qda_ds
caret_qda_ds <- train(y~.,data=ds_data_for,method = "qda",
                        trControl = trainControl(method = "CV",number = 10))
caret_qda_ds$results
cv_qda_ds <- qda(y~.,data=ds_data_for,CV = TRUE)
err_qda_ds <- confusionMatrix(reference = as.factor(ds_data_for$y),data = cv_qda_ds$class)
err_qda_ds

QDA<-qda(y~.,data=data_for)

pred <- predict(QDA, newdata =test_data)
pred
set.seed(1)
caret_qda <- train(y~.,data=data_for,method = "qda",
                   trControl = trainControl(method = "CV",number = 10))
caret_qda$results
cv_qda <- qda(y~.,data=data_for,CV = TRUE)
err <- confusionMatrix(reference = as.factor(data_for$y),data = cv_qda$class)
err

##LDA
## Best subset
LDA<-lda(y~.,data=data_for)
pred_lda <- predict(LDA, newdata =test_data)
caret_lda <- train(y~.,data=data_for,method = "lda",
                   trControl = trainControl(method = "CV",number = 10))
caret_lda$results
cv_lda <- lda(y~.,data=data_for,CV = TRUE)
err_lda <- confusionMatrix(reference = as.factor(data_for$y),data = cv_lda$class)
err_lda

##Downsampled best subset
LDA_ds<-lda(y~.,data=ds_data_for)
pred_lda_ds <- predict(LDA_ds, newdata =test_data)
caret_lda_ds <- train(y~.,data=ds_data_for,method = "lda",
                   trControl = trainControl(method = "CV",number = 10))
caret_lda_ds$results
cv_lda_ds <- lda(y~.,data=ds_data_for,CV = TRUE)
err_lda_ds <- confusionMatrix(reference = as.factor(ds_data_for$y),data = cv_lda_ds$class)
err_lda_ds

##NB
install.packages('klaR')
library(klaR)
## NB on full trainign data with best preds
nb <- NaiveBayes(y~.,data=data_for,usekernel = FALSE)
nb_kern <- NaiveBayes(y~.,data=data_for,usekernel = TRUE)
pred_nb<- predict(nb, newdata =test_data)
pred_nb_kern <- predict(nb_kern, newdata =test_data)
caret_nb <- train(y~.,data=data_for,method = "nb",
                   trControl = trainControl(method = "CV",number = 10))
caret_nb$results
## NB on downsampled data with best preds
nb <- NaiveBayes(y~.,data=data_for,usekernel = FALSE)
nb_kern <- NaiveBayes(y~.,data=data_for,usekernel = TRUE)
pred_nb<- predict(nb, newdata =test_data)
pred_nb_kern <- predict(nb_kern, newdata =test_data)
caret_nb_ds <- train(y~.,data=ds_data_for,method = "nb",
                  trControl = trainControl(method = "CV",number = 10))
caret_nb_ds$results
