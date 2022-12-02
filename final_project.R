#ST 563 Final project

library(keras)
library(tensorflow)
library(rsample)
library (dplyr)
library(corrplot)
library(dplyr)
library (tfruns)
library (glmnet)
library(caret)
library(h2o)

Train <- read.csv("YearPredictTrain.txt", stringsAsFactors=FALSE)
Test <- read.csv("YearPredictTest.txt", stringsAsFactors=FALSE)

head(Train)
head(Test)
ncol(Train)

####################### DIMENSION REDUCTION ###########################3

#### Applying PCA for dimension reduction
#Extract only predictors and scale them
set.seed(1001)
X<-scale(Train[,0:90], center=TRUE, scale=FALSE)
dim(X)
head(X)
X_test<-scale(Test[,0:90], center=TRUE, scale=TRUE)
dim(X_test)
#Total variation
TV=sum(apply(X,2,var))
apply(X,2,var)
#Standardized predictors
Xstd<-scale(X, center=TRUE, scale=TRUE)
#TV
TV=ncol(Xstd)
#PCA
pc_out<-prcomp(Xstd)
names(pc_out)
#PCs
Z<-pc_out$x
dim(Z)
sum(apply(Z,2,var))
#Proportion of TV captured by PC1
var(Z[,1])/TV
# Cumulative proportion of TV captured by successive PCS
cumsum(apply(Z, 2, var)) / TV
summary(pc_out)
#loadings
loadings<-pc_out$rotation
round(loadings[,1],2)

###PCR
library(pls)
set.seed(1001)
pcr_lm <- pcr(Year ~ timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg4+timbreAvg5+timbreAvg6+timbreAvg7+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov14+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov19+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov30+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov37+timbreCov38+timbreCov39+timbreCov40+timbreCov41+timbreCov42+timbreCov43+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov54+timbreCov55+timbreCov56+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov65+timbreCov66+timbreCov67+timbreCov68+timbreCov69+timbreCov70+timbreCov71+timbreCov72+timbreCov73+timbreCov74+timbreCov75+timbreCov76+timbreCov77+timbreCov78,
              data = Train,
              center = TRUE, scale = TRUE,
              validation = "CV")
summary(pcr_lm)

#refit PCR and performing CV using caret
set.seed(1001)
model <- train(Year~timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg4+timbreAvg5+timbreAvg6+timbreAvg7+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov14+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov19+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov30+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov37+timbreCov38+timbreCov39+timbreCov40+timbreCov41+timbreCov42+timbreCov43+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov54+timbreCov55+timbreCov56+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov65+timbreCov66+timbreCov67+timbreCov68+timbreCov69+timbreCov70+timbreCov71+timbreCov72+timbreCov73+timbreCov74+timbreCov75+timbreCov76+timbreCov77+timbreCov78,
               data = Train,
               method = "pcr",
               trControl = trainControl("cv", number = 10),
               tuneLength = 50,
               preProcess = c("center", "scale"))
model$results
SE <- model$results$RMSESD/sqrt(10)
round(SE, 2)

pcr_final <- pcr(Year~timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg4+timbreAvg5+timbreAvg6+timbreAvg7+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov14+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov19+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov30+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov37+timbreCov38+timbreCov39+timbreCov40+timbreCov41+timbreCov42+timbreCov43+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov54+timbreCov55+timbreCov56+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov65+timbreCov66+timbreCov67+timbreCov68+timbreCov69+timbreCov70+timbreCov71+timbreCov72+timbreCov73+timbreCov74+timbreCov75+timbreCov76+timbreCov77+timbreCov78,
                 data = Train,
                 center = TRUE, scale = TRUE,
                 ncomp = 50, validation = "none")
summary(pcr_final)

############PLS
set.seed(1001)
model <- train(Year~timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg4+timbreAvg5+timbreAvg6+timbreAvg7+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov14+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov19+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov30+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov37+timbreCov38+timbreCov39+timbreCov40+timbreCov41+timbreCov42+timbreCov43+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov54+timbreCov55+timbreCov56+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov65+timbreCov66+timbreCov67+timbreCov68+timbreCov69+timbreCov70+timbreCov71+timbreCov72+timbreCov73+timbreCov74+timbreCov75+timbreCov76+timbreCov77+timbreCov78,
               data = Train,
               preProcess = c("center", "scale"),
               method = "pls",
               trControl = trainControl("cv", number = 10),
               tuneLength = 50
)
model

pls_final <- plsr(Year~timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg4+timbreAvg5+timbreAvg6+timbreAvg7+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov14+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov19+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov30+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov37+timbreCov38+timbreCov39+timbreCov40+timbreCov41+timbreCov42+timbreCov43+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov54+timbreCov55+timbreCov56+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov65+timbreCov66+timbreCov67+timbreCov68+timbreCov69+timbreCov70+timbreCov71+timbreCov72+timbreCov73+timbreCov74+timbreCov75+timbreCov76+timbreCov77+timbreCov78,
                  data = Train,
                  center = TRUE, scale = TRUE,
                  ncomp = 29)
summary(pls_final)
pls_scores <- scores(pls_final)
pls_scores
load<-loadings(pls_final)
load


############################### VARIABLE SELECTION ###########################
#################### forward selection
library(leaps)

forward <- regsubsets(Year ~ timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg4+timbreAvg5+timbreAvg6+timbreAvg7+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov14+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov19+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov30+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov37+timbreCov38+timbreCov39+timbreCov40+timbreCov41+timbreCov42+timbreCov43+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov54+timbreCov55+timbreCov56+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov65+timbreCov66+timbreCov67+timbreCov68+timbreCov69+timbreCov70+timbreCov71+timbreCov72+timbreCov73+timbreCov74+timbreCov75+timbreCov76+timbreCov77+timbreCov78,
                      data = Train,
                      nvmax = 90,
                      method = "forward")
# summary
mod_summary <- summary(forward)
mod_summary

#get aic, bic, adjrsquare
metrics <- data.frame(aic = mod_summary$cp,
                      bic = mod_summary$bic,
                      adjR2 = mod_summary$adjr2)
metrics

round( coef(forward, 72), 4)


############Using the holdout and Cross-Validation for forward selection
set.seed(1001)
Test<-Test[,-92]
Test

## Best subset selection on the training data
forward <- regsubsets(Year ~ timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg4+timbreAvg5+timbreAvg6+timbreAvg7+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov14+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov19+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov30+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov37+timbreCov38+timbreCov39+timbreCov40+timbreCov41+timbreCov42+timbreCov43+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov54+timbreCov55+timbreCov56+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov65+timbreCov66+timbreCov67+timbreCov68+timbreCov69+timbreCov70+timbreCov71+timbreCov72+timbreCov73+timbreCov74+timbreCov75+timbreCov76+timbreCov77+timbreCov78,
                      data = Train,
                      nvmax = 90,
                      method = "forward")
train_sum <- summary(forward)

#for each model size, estimate test performance
test_err <- function(mod_size,
                     reg_summary,
                     test_model,
                     test_resp){
  #get regression coefs
  betahat <- coef(reg_summary$obj, mod_size)
  # get forward subset of the specified size
  sub <- reg_summary$which[mod_size, ]
  # Create test model matrix, prediction, test error
  model <- test_model[, sub]
  yhat <- model %*% betahat
  err <- mean((test_resp - yhat)^2)
  return(err)
  
  }
## Apply the function above to each model size
test_model <- model.matrix(Year~timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg4+timbreAvg5+timbreAvg6+timbreAvg7+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov14+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov19+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov30+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov37+timbreCov38+timbreCov39+timbreCov40+timbreCov41+timbreCov42+timbreCov43+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov54+timbreCov55+timbreCov56+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov65+timbreCov66+timbreCov67+timbreCov68+timbreCov69+timbreCov70+timbreCov71+timbreCov72+timbreCov73+timbreCov74+timbreCov75+timbreCov76+timbreCov77+timbreCov78,
                           data = Train)
test_resp <- Train$Year
hold_err <- sapply(1:90, test_err,
                   reg_summary = train_sum,
                   test_model = test_model,
                   test_resp = test_resp)
plot(hold_err, type = 'b', pch=19, lwd=2)

## Best model size and refit of full data
size_opt <- which.min(hold_err)
bestmod <- regsubsets(Year~timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg4+timbreAvg5+timbreAvg6+timbreAvg7+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov14+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov19+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov30+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov37+timbreCov38+timbreCov39+timbreCov40+timbreCov41+timbreCov42+timbreCov43+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov54+timbreCov55+timbreCov56+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov65+timbreCov66+timbreCov67+timbreCov68+timbreCov69+timbreCov70+timbreCov71+timbreCov72+timbreCov73+timbreCov74+timbreCov75+timbreCov76+timbreCov77+timbreCov78,
                      data = Train,
                      nvmax = 90,
                      method = "forward")
coef(bestmod, size_opt)
bestmod

############################ REGRESSION MODELS ######################
############################## Random forest
library(randomForest)
set.seed(1001)
randf <- randomForest(Year~timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg5+timbreAvg6+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov38+timbreCov40+timbreCov41+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov66+timbreCov69+timbreCov70+timbreCov71+timbreCov73+timbreCov75+timbreCov76+timbreCov77,
                      data = Train,
                      mtry = 10,
                      importance = TRUE)
print(randf)
# Number of trees: 500
#No. of variables tried at each split: 10
#Mean of squared residuals: 86.9076
#% Var explained: 26.79

newx<-Test[,1:91]
ytest<-Test['Year']
pred<-predict(randf, newdata=newx)
pred

randf_train_MSE<-mean((Train$Year - pred)^2)
randf_train_MSE

randf_test_MSE<-mean((Test$Year - pred)^2)
randf_test_MSE


########################## SVM Regresssion
#Tuning hyperparamaters for SVM with kernel=radial
library(e1071)
set.seed(1)

#tuning cost 
tune.out <- tune(svm , Train$Year~timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg5+timbreAvg6+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov38+timbreCov40+timbreCov41+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov66+timbreCov69+timbreCov70+timbreCov71+timbreCov73+timbreCov75+timbreCov76+timbreCov77,
                 data = Train,
                 kernel = "radial",
                 ranges = list(
                   cost = c(0.1 , 1, 10, 100, 1000) ,
                   gamma = c(0.5, 1, 2, 3, 4)
                 )
)
summary(tune.out)

#Perform SVM with best tuned cost & gamma
#Using the data selected by forward selection 
#Train the model
svr <- svm(Year~timbreAvg1+timbreAvg2+timbreAvg3+timbreAvg5+timbreAvg6+timbreAvg8+timbreAvg9+timbreAvg10+timbreAvg11+timbreAvg12+timbreCov1+timbreCov2+timbreCov3+timbreCov4+timbreCov5+timbreCov6+timbreCov7+timbreCov8+timbreCov9+timbreCov10+timbreCov11+timbreCov12+timbreCov13+timbreCov15+timbreCov16+timbreCov17+timbreCov18+timbreCov20+timbreCov21+timbreCov22+timbreCov23+timbreCov24+timbreCov25+timbreCov26+timbreCov27+timbreCov28+timbreCov29+timbreCov31+timbreCov32+timbreCov33+timbreCov34+timbreCov35+timbreCov36+timbreCov38+timbreCov40+timbreCov41+timbreCov44+timbreCov45+timbreCov46+timbreCov47+timbreCov48+timbreCov49+timbreCov50+timbreCov51+timbreCov52+timbreCov53+timbreCov57+timbreCov58+timbreCov59+timbreCov60+timbreCov61+timbreCov62+timbreCov63+timbreCov64+timbreCov66+timbreCov69+timbreCov70+timbreCov71+timbreCov73+timbreCov75+timbreCov76+timbreCov77,
           data = Train,
           type = "eps-regression",
           kernel = "radial", gamma = 1,
           cost = 1, epsilon = 0.1
)
summary(svr)

#No. of support vectors=83280 at cost=1, epsilon=0.1,gamma=1
#No. of support vectors=81507 at cost=1000, epsilon=0.1,gamma=1
# Prediction
X_Test<-Test[,1:90]
pred_svr <- predict(svr, newdata = X_Test)

svr_train_MSE<-mean((Train$Year - pred_svr)^2)
svr_train_MSE
#120.7677 at cost 1
#118.8391 at cost 1000
svr_test_MSE<-mean((Test$Year - pred_svr)^2)
svr_test_MSE
#121.7005 at cost 1
#119.6802 at cost 1000


############ STACKING / ENSEMBLE METHODS ########################

#Get response and feature names
Y<-"Year"
X<-setdiff(names(Train),Y)

#Init h2o
h2o.init()
# Convert train/test sets to h2o format
Train <- as.h2o(Train)
Test <- as.h2o(Test)

nfolds <- 3
seed <- 1001
# Lasso
stack_glm <- h2o.glm(
  x = X, y = Y, training_frame = Train,
  nfolds = nfolds, seed = seed,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Modulo",
  alpha = 1, remove_collinear_columns = TRUE
)
# Random forest
stack_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = Train,
  nfolds = nfolds, seed = seed,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Modulo",
  ntrees = 200, mtries = 5, max_depth = 15,
  sample_rate = 0.8, stopping_rounds = 50,
  stopping_metric = "RMSE", stopping_tolerance = 0
)
# GBM
stack_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = Train,
  nfolds = nfolds, seed = seed,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Modulo",
  ntrees = 5000, learn_rate = 0.01,
  max_depth = 7, min_rows = 5, sample_rate = 0.8,
  stopping_rounds = 50, stopping_metric = "RMSE",
  stopping_tolerance = 0
)
stack_ensemble <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = Train,
  model_id = "stack_ensemble",
  base_models = list(stack_glm, stack_rf, stack_gbm),
  metalearner_algorithm = "AUTO"
)
# Function to extract rmse
get_rmse <- function(model){
  perf <- h2o.performance(model,
                          newdata = Test)
  return(perf@metrics$RMSE)
}
# model list
mod_list <- list(GLM = stack_glm, RF = stack_rf,
                 GBM = stack_gbm, STACK = stack_ensemble)
# apply function to model list
purrr::map_dbl(mod_list, get_rmse)
#GLM RMSE=9.589912
#RF RMSE=9.840460
#GBM RMSE=9.091583
#Stack RMSE= 10.929397

pred <- data.frame(
  GLM_pred = as.vector(h2o.getFrame(hit_glm@model$cross_validation_holdout_predictions_frame_id$name)),
  RF_pred = as.vector(h2o.getFrame(hit_rf@model$cross_validation_holdout_predictions_frame_id$name)),
  GBM_pred = as.vector(h2o.getFrame(hit_gbm@model$cross_validation_holdout_predictions_frame_id$name))
)
cor(pred)
# cor(pred)
#GLM_pred   RF_pred  GBM_pred
#GLM_pred 1.0000000 0.7794983 0.8056723
#RF_pred  0.7794983 1.0000000 0.8563329
#GBM_pred 0.8056723 0.8563329 1.0000000
plot(pred)



