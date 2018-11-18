rm(list=ls(all=TRUE))
library(caret)
library(plyr)
library(ada)
set.seed(123)
library(AppliedPredictiveModeling)
library(ROCR)


data_num = read.csv("German_All_Numerical.csv", header = T)
data_num$V21 <- as.character(data_num$V21)
data_num$V21 <- revalue(data_num$V21, c("2"="0"))
data_num$V21 = as.numeric(data_num$V21)
sample_train<- sample(seq_len(nrow(data_num)), size = floor(0.75*nrow(data_num)))
sample_test <- sample(seq_len(nrow(data_num)), size = floor(0.25*nrow(data_num)))
#sample_validation <- sample(seq_len(nrow(data_num)), size = floor(0.20*nrow(data_num)))
train     <- data_num[sample_train, ]
test      <- data_num[sample_test, ]
#validation  <- data_num[sample_validation, ]
train$V21 =  as.factor(train$V21)
test$V21 =  as.factor(test$V21)
#validation$V21 =  as.factor(validation$V21)

fitControl <- trainControl(method = "repeatedcv", number = 4, repeats = 4)

modelFitGLM <- train(V21 ~ ., data = train, method = "glm")
modelFitADA <- ada(V21 ~., data = train , iter = 20, loss="logistic")
modelFitSVM <- train(V21 ~ ., data = train, method = "svmLinear")
modelFitGBM <- train(as.factor(V21) ~ ., data = train, method = "gbm", trControl = fitControl,verbose = FALSE)



#predGLM <- predict(modelFitGLM, newdata = validation)
#predADA <- predict(modelFitADA, newdata = validation)
#prefSVM <- predict(modelFitSVM, newdata = validation)
#predDF <- data.frame(predGLM, predADA, prefSVM, V21 = validation$V21, stringsAsFactors = F)

#modelStack <- train(V21 ~ ., data = predDF, method = "glm")

testPredGLM <- predict(modelFitGLM, newdata = test)
testPredADA <- predict(modelFitADA, newdata = test)
testPredSVM <- predict(modelFitSVM, newdata = test)


#testPredLevelOne <- data.frame(testPredGLM, testPredADA, testPredSVM, V21 = test$V21, stringsAsFactors = F)
#combPred <- predict(modelStack, testPredLevelOne)
#confusionMatrix(combPred, test$V21)$overall[1]

confusionMatrix(testPredGLM, test$V21)$overall[1]
confusionMatrix(testPredADA, test$V21)$overall[1]
confusionMatrix(testPredSVM, test$V21)$overall[1]

