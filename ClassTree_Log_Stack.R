
train = read.delim(file = "data/YearPredictTrain.txt", header = TRUE, sep = ",", dec = ".")
test = read.delim(file = "data/YearPredictTest.txt", header = TRUE, sep = ",", dec = ".")

head(train)
summary(train)

library(corrplot)
library(dplyr)
library(caret)
library(groupdata2)
# corrplot(cor(select (train, -c (Year, Class))), method = "circle", tl.pos='n')
# 
# 
# pc = prcomp(select (train, -c (Year, Class)),
#             center = TRUE,
#             scale. = TRUE)
# summary(pc)
# 
# 
# sum(is.na(train))
# sum(is.na(test))



set.seed(5)

train_cl = select (train, -c ( Year))
train_cl$Class = as.factor(train_cl$Class)
train_cl_down <- downsample(train_cl, cat_col = "Class", id_method = "n_ids")

summary(train_cl$Class)
summary(train_cl_down$Class)

test_cl = select (test, -c ( Year))
test_cl$Class = as.factor(test_cl$Class)
test_cl_down <- downsample(test_cl, cat_col = "Class", id_method = "n_ids")

summary(test_cl$Class)
summary(test_cl_down$Class)



train_control = trainControl(method = "cv", 
                             number = 5, 
                             search = "grid",
                             summaryFunction = multiClassSummary,
                             seeds = set.seed(50))


#Classification Tree
library(rpart)
library(rpart.plot)

#For full data
ct_model_cl <- rpart(Class ~ .,
                     data = train_cl,
                     method='class',
                     parms = list(split = "information"),
                     control = rpart.control(xval = 10,
                                             minbucket = 2,
                                             cp = 0))
printcp(ct_model_cl)

cp <- ct_model_cl$cptable
row_min <- which.min(cp[,4])
row_min
cp[row_min,4]+cp[row_min,5]

row_1se <- which(cp[,4]<cp[row_min,4]+cp[row_min,5])[1]

ct_final_cl <- prune(ct_model_cl, cp = cp[row_1se,1])
rpart.plot(ct_final_cl)

pred_ct_cl <- predict(ct_final_cl, test_cl,type = 'class')
confusionMatrix (pred_ct_cl, test_cl$Class)
#accuracy 0.6326


#For downsampled data
ct_model_cl <- rpart(Class ~ .,
                     data = train_cl_down,
                     method='class',
                     parms = list(split = "information"),
                     control = rpart.control(xval = 10,
                                             minbucket = 2,
                                             cp = 0))
printcp(ct_model_cl)

cp <- ct_model_cl$cptable
row_min <- which.min(cp[,4])
row_min
cp[row_min,4]+cp[row_min,5]

row_1se <- which(cp[,4]<cp[row_min,4]+cp[row_min,5])[1]

ct_final_cl <- prune(ct_model_cl, cp = cp[row_1se,1])
rpart.plot(ct_final_cl)

pred_ct_cl_down <- predict(ct_final_cl, test_cl_down,type = 'class')
confusionMatrix (pred_ct_cl_down, test_cl_down$Class)
#accuracy 0.5136



#Logistic Regression
library(nnet)

# For full data
# multinomial logistic regression
multilogit <- multinom(Class ~ .,
                       data = train_cl,
                       methods = 'class',
                       maxit = 200, trace=FALSE)
# summary
summary(multilogit)
head(multilogit$fitted.values)


pred_log_cl <- predict(multilogit, test_cl,type = 'class')
confusionMatrix (pred_log_cl, test_cl$Class)
#accuracy 0.6639




# For downsampled data
# multinomial logistic regression
multilogit <- multinom(Class ~ .,
                       data = train_cl_down,
                       methods = 'class',
                       maxit = 200, trace=FALSE)
# summary
summary(multilogit)
head(multilogit$fitted.values)


pred_log_cl_down <- predict(multilogit, test_cl_down,type = 'class')
confusionMatrix (pred_log_cl_down, test_cl_down$Class)
#accuracy 0.6125 






#Stacking

Y <- "Class"
X <- setdiff(names(train_cl), Y)

#For full data
# Init h2o
h2o.init()
# Convert train/test sets to h2o format
train_h2o <- as.h2o(train_cl)
test_h2o <- as.h2o(test_cl)


nfolds <- 5
seed <- 123
# # log
# hit_glm <- h2o.glm(
#   x = X, y = Y, training_frame = train_h2o,
#   nfolds = nfolds, seed = seed,
#   family = 'binomial',
#   keep_cross_validation_predictions = TRUE,
#   fold_assignment = "Modulo",
#   alpha = 1, remove_collinear_columns = TRUE
# )

#Random Forest 
hit_rf <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o,
  nfolds = nfolds, seed = seed,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Modulo",
  ntrees = 50, max_depth = 20,
  sample_rate = 0.632, stopping_rounds = 50,
  stopping_metric = "misclassification", stopping_tolerance = 0
)


# GBM
hit_gbm <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o,
  nfolds = nfolds, seed = seed,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Modulo",
  ntrees = 50, learn_rate = 0.1,
  max_depth = 7, min_rows = 10, sample_rate = 0.632,
  stopping_rounds = 50, stopping_metric = "misclassification",
  stopping_tolerance = 0
)


hit_ensemble <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o,
  model_id = "class_ensemble",
  base_models = list(hit_rf, hit_gbm),
  metalearner_algorithm = "AUTO"
)



table_rf <- h2o.confusionMatrix(hit_rf,newdata=test_h2o)
table_rf
1-table_rf[,4] #Sensitivity

c(sum(table_rf[2:3,2:3])/sum(table_rf[2:3,1:3]),
  sum(table_rf[c(1,3),c(1,3)])/sum(table_rf[c(1,3),1:3]),
  sum(table_rf[1:2,1:2])/sum(table_rf[1:2,1:3])) #Specificity

1-table_rf[4,4] #Accuracy

#Accuracy0.6524

table_gbm <- h2o.confusionMatrix(hit_gbm,newdata=test_h2o)
table_gbm
1-table_gbm[,4] #Sensitivity

c(sum(table_gbm[2:3,2:3])/sum(table_gbm[2:3,1:3]),
  sum(table_gbm[c(1,3),c(1,3)])/sum(table_gbm[c(1,3),1:3]),
  sum(table_gbm[1:2,1:2])/sum(table_gbm[1:2,1:3])) #Specificity

1-table_gbm[4,4] #Accuracy
#Accuracy0.6746

table_ensemble <- h2o.confusionMatrix(hit_ensemble,newdata=test_h2o)
table_ensemble
1-table_ensemble[,4] #Sensitivity

c(sum(table_ensemble[2:3,2:3])/sum(table_ensemble[2:3,1:3]),
  sum(table_ensemble[c(1,3),c(1,3)])/sum(table_ensemble[c(1,3),1:3]),
  sum(table_ensemble[1:2,1:2])/sum(table_ensemble[1:2,1:3])) #Specificity

1-table_ensemble[4,4]
#Accuracy0.6752







#For downsampled data
# Init h2o
h2o.init()
# Convert train/test sets to h2o format
train_h2o_down <- as.h2o(train_cl_down)
test_h2o_down <- as.h2o(test_cl_down)


nfolds <- 5
seed <- 123

#Random Forest 
hit_rf_down <- h2o.randomForest(
  x = X, y = Y, training_frame = train_h2o_down,
  nfolds = nfolds, seed = seed,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Modulo",
  ntrees = 50, max_depth = 20,
  sample_rate = 0.632, stopping_rounds = 50,
  stopping_metric = "misclassification", stopping_tolerance = 0
)


# GBM
hit_gbm_down <- h2o.gbm(
  x = X, y = Y, training_frame = train_h2o_down,
  nfolds = nfolds, seed = seed,
  keep_cross_validation_predictions = TRUE,
  fold_assignment = "Modulo",
  ntrees = 50, learn_rate = 0.1,
  max_depth = 7, min_rows = 10, sample_rate = 0.632,
  stopping_rounds = 50, stopping_metric = "misclassification",
  stopping_tolerance = 0
)


hit_ensemble_down <- h2o.stackedEnsemble(
  x = X, y = Y, training_frame = train_h2o_down,
  model_id = "class_ensemble",
  base_models = list(hit_rf_down, hit_gbm_down),
  metalearner_algorithm = "AUTO"
)



table_rf_down <- h2o.confusionMatrix(hit_rf_down,newdata=test_h2o_down)
table_rf_down
1-table_rf_down[4,4] #Accuracy
1-table_rf_down[,4] #Sensitivity
c(sum(table_rf_down[2:3,2:3])/sum(table_rf_down[2:3,1:3]),
  sum(table_rf_down[c(1,3),c(1,3)])/sum(table_rf_down[c(1,3),1:3]),
  sum(table_rf_down[1:2,1:2])/sum(table_rf_down[1:2,1:3])) #Specificity
#Accuracy0.5959471

table_gbm_down <- h2o.confusionMatrix(hit_gbm_down,newdata=test_h2o_down)
table_gbm_down
1-table_gbm_down[4,4]#Accuracy
1-table_gbm_down[,4]#Sensitivity
c(sum(table_gbm_down[2:3,2:3])/sum(table_gbm_down[2:3,1:3]),
  sum(table_gbm_down[c(1,3),c(1,3)])/sum(table_gbm_down[c(1,3),1:3]),
  sum(table_gbm_down[1:2,1:2])/sum(table_gbm_down[1:2,1:3])) #Specificity
#Accuracy0.6162117




table_ensemble_down <- h2o.confusionMatrix(hit_ensemble_down,newdata=test_h2o_down)
table_ensemble_down
1-table_ensemble_down[,4]#sensitivity

c(sum(table_ensemble_down[2:3,2:3])/sum(table_ensemble_down[2:3,1:3]),
  sum(table_ensemble_down[c(1,3),c(1,3)])/sum(table_ensemble_down[c(1,3),1:3]),
  sum(table_ensemble_down[1:2,1:2])/sum(table_ensemble_down[1:2,1:3])) #Specificity

1-table_ensemble_down[4,4]
#Accuracy0.6236559






















#boost Logistic Regression
# logGrid <- expand.grid(
#   iter = seq(500, 2000, by=300)
#   )
# 
# log_model_cl = caret::train( 
#   x  = select(train_cl, -c ("Class")),
#   y = train_cl$Class,
#   trControl = train_control,
#   method = "LMT",
#   tuneGrid = logGrid)
# 
# pred_log_cl <- predict(log_model_cl, test_cl, type='class')
# 
# confusionMatrix (pred_log_cl, test_cl$Class)
# 
# #Boosted Clasification Tree
# 
# # bctGrid <- expand.grid(
# #   iter = seq(500, 2000, by=300),
# #   maxdepth = c(1, 2, 3),
# #   nu = c(0.3, 0.1, 0.05, 0.01, 0.005)
# #   )
# 
# 
# 
# bct_model_cl = caret::train( 
#   x  = select(train_cl, -c ("Class")),
#   y = train_cl$Class,
#   trControl = train_control,
#   method = "ada",
#   tuneGrid = bctGrid)
# 
# pred_bct_cl <- predict(bct_model_cl, test_cl)
# 
# confusionMatrix (pred_xgboost_cl, test_xgb_cl$Class)
# 
# 
# 
# 
# 
# 
# 
# #AdaBoost Classification Trees
# abctGrid <- expand.grid(
#   iter = seq(500, 2000, by=300),
#   maxdepth = c(1, 2, 3), 
#   nu = c(0.3, 0.1, 0.05, 0.01, 0.005)
# )
# 
# 
# bct_model_cl = caret::train( 
#   x  = select (train_cl, -c ("Class")),
#   y = train_cl$Class,
#   trControl = train_control,
#   method = "adaboost",
#   tuneGrid = bctGrid)