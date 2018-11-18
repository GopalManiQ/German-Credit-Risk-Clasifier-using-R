rm(list=ls(all=TRUE))
setwd('C:\\Users\\Gopal Mukkamala\\Desktop\\Credit Risk')
#library(car)
library(caTools)
library(corrplot)
library(caret)
library(plyr)
library(ROCR)
library(e1071)
library(ada)
library(ggplot2)
library(pROC)


predictRisk <- function(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20){

test_sample = data.frame(c(V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20))  
  
#Importing Datasets
#data_categorical <- read.table("german.data", sep = "" , header = F , nrows = 1000, na.strings = "",stringsAsFactors= F)
data_numerical <- read.csv("German_All_Numerical.csv", header = T)


#Preparing Data
data_numerical$V21 <- as.character(data_numerical$V21)
data_numerical$V21 <- revalue(data_numerical$V21, c("2"="0"))
data_numerical$V21 = as.numeric(data_numerical$V21)
data_numerical$X = NULL


#Normalization
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

data_normalized <- as.data.frame(lapply(data_numerical, normalize))
data_numerical <- data_normalized


#No. of NA's per column
colSums(is.na(data_numerical))


#Outlier Detection and Treatment
#OutlierVals = boxplot(data_numerical,col = "red")$out
#source("OutlierKD.R")
#outlierKD(data_numerical,V5)

##PCA - Principle Component Analysis

#Removing the Response variable
PCA_Data = subset(data_numerical, select = -V21)
str(PCA_Data)

#Splitting into train&test
sample_train_PCA<- sample(seq_len(nrow(PCA_Data)), size = floor(0.75*nrow(PCA_Data)))
sample_test_PCA <- sample(seq_len(nrow(PCA_Data)), size = floor(0.25*nrow(PCA_Data)))
PCA_train     <- PCA_Data[sample_train_PCA, ]
PCA_test      <- PCA_Data[sample_test_PCA, ]

#PCA
PCA_result <- prcomp(PCA_train, scale. = T)
names(PCA_result)
PCA_result$center
PCA_result$scale
PCA_result$rotation
biplot(PCA_result,scale = 0)

#Standard Deviation and Variance
std_dev = PCA_result$sdev
PCA_var = std_dev^2
prop_var <- PCA_var/sum(PCA_var)
prop_var[1:20]

#Plot1
plot(prop_var, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained",
     type = "b")

#Plot2
plot(cumsum(prop_var), xlab = "Principal Component",
     ylab = "Cumulative Proportion of Variance Explained",
     type = "b")

#Reducing the highly correlated attributes
df_cor_numerical <- cor(data_numerical[1:(ncol(data_numerical)-1)], use = "complete.obs")
corrplot(df_cor_numerical, type="lower")

#Removing highly correlated attributes
ReduceCorrelated <- findCorrelation(df_cor_numerical, cutoff=0.8, verbose = T)

##No highly correlated Values


#Splitting dataset into Train & Test datasets
sample_train <- sample(seq_len(nrow(data_numerical)), size = floor(0.75*nrow(data_numerical)))
sample_test <- sample(seq_len(nrow(data_numerical)), size = floor(0.25*nrow(data_numerical)))
train     <- data_numerical[sample_train, ]
test      <- data_numerical[sample_test, ]
str(train)
str(test)

#Convering Response variable to factor type for classification
train$V21 = as.factor(train$V21)
test$V21 = as.factor(test$V21)

#NNET metrics
fitControl <- trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)


trainNNET <- train
testNNET <- test

trainNNET$V21 = as.factor(trainNNET$V21)
testNNET$V21 = as.factor(testNNET$V21)


#Convering Response variables to good/bad(only for NNET model)
trainNNET$V21 = ifelse(trainNNET$V21 == 0, "bad","good")
testNNET$V21 = ifelse(testNNET$V21 == 0, "bad","good")


#Training Models
modelFitGLM <- train(V21 ~ ., data = train, method = "glm")
modelFitADA <- ada(V21 ~., data = train , iter = 20, loss="logistic")
modelFitSVM <- train(V21 ~ ., data = train, method = "svmLinear")
modelFitRF <- train(V21 ~ ., data = train, method = "rf",ntree = 5)
modelFitNNET <- train(V21 ~ ., data = trainNNET, method = "nnet", metric = "ROC", trControl = fitControl,verbose = FALSE)



#Summary of Models
summary(modelFitGLM)
summary(modelFitADA)
summary(modelFitSVM)
summary(modelFitRF)
summary(modelFitNNET)


#Test Prediction 
testPredGLM <- predict(modelFitGLM, newdata = test)
testPredADA <- predict(modelFitADA, newdata = test)
testPredSVM <- predict(modelFitSVM, newdata = test)
testPredRF <- predict(modelFitRF, newdata = test)
testPredNNET <- predict(modelFitNNET, newdata = testNNET)



result <- predict(modelFitADA, newdata = test_sample)

return(result)


}


#Confusion Matrices
confusionMatrix(testPredGLM, test$V21)
confusionMatrix(testPredADA, test$V21)
confusionMatrix(testPredSVM, test$V21)
confusionMatrix(testPredRF, test$V21)
confusionMatrix(testPredNNET, testNNET$V21)


#ROC Curves
#Logistic
pred = prediction(as.numeric(testPredGLM),as.numeric(test$V21))
perfGLM = performance(pred,"tpr","fpr")
plot(perfGLM,type = "b",colorize = T,lwd = 3)

#AdaBoost
pred = prediction(as.numeric(testPredADA),as.numeric(test$V21))
perfADA = performance(pred,"tpr","fpr")
plot(perfADA,type = "b",colorize = T, add = T,lwd = 3)


#Support Vector Machine
pred = prediction(as.numeric(testPredSVM),as.numeric(test$V21))
perfSVM = performance(pred,"tpr","fpr")
plot(perfSVM,colorize = T, add = T,lwd = 3)

#Random Forest
pred = prediction(as.numeric(testPredRF),as.numeric(test$V21))
perfRF = performance(pred,"tpr","fpr")
plot(perfRF,colorize = T, add = T,lwd = 3)

#Neural Network
pred = prediction(as.numeric(testPredNNET),as.numeric(testNNET$V21))
perfNN = performance(pred,"tpr","fpr",lwd = 3,add = T)
plot(perfNN,colorize = T,add = T, lwd = 3)







library(shiny)

ui <- fluidPage(
  
  titlePanel("Input all 20 attributes to predict Credit Risk"),
  
  fluidRow(
    
    
    tags$head(
      tags$style(type="text/css", "#inline label{ display: table-cell; text-align: center; vertical-align: middle; } 
                 #inline .form-group { display: table-row;}")
      ),
    
    tags$div(id = "inline", textInput(inputId = "txtInp1", label = "Status of existing checking account:")),
    tags$div(id = "inline", textInput(inputId = "txtInp2", label = "Duration in month:")),
    tags$div(id = "inline", textInput(inputId = "txtInp3", label = "Credit history:")),
    tags$div(id = "inline", textInput(inputId = "txtInp4", label = "Purpose:")),
    tags$div(id = "inline", textInput(inputId = "txtInp5", label = "Credit amount:")),
    tags$div(id = "inline", textInput(inputId = "txtInp6", label = "Savings account/bonds:")),
    tags$div(id = "inline", textInput(inputId = "txtInp7", label = "Present employment since:")),
    tags$div(id = "inline", textInput(inputId = "txtInp8", label = "Installment rate in percentage of disposable income:")),
    tags$div(id = "inline", textInput(inputId = "txtInp9", label = "Personal status and sex:")),
    tags$div(id = "inline", textInput(inputId = "txtInp10", label = "Other debtors / guarantors:")),
    tags$div(id = "inline", textInput(inputId = "txtInp11", label = "Present residence since:")),
    tags$div(id = "inline", textInput(inputId = "txtInp12", label = "Property:")),
    tags$div(id = "inline", textInput(inputId = "txtInp13", label = "Age in years:")),
    tags$div(id = "inline", textInput(inputId = "txtInp14", label = "Other installment plans :")),
    tags$div(id = "inline", textInput(inputId = "txtInp15", label = "Housing:")),
    tags$div(id = "inline", textInput(inputId = "txtInp16", label = "Number of existing credits at this bank:")),
    tags$div(id = "inline", textInput(inputId = "txtInp17", label = "Job:")),
    tags$div(id = "inline", textInput(inputId = "txtInp18", label = "Number of people being liable to provide maintenance for:")),
    tags$div(id = "inline", textInput(inputId = "txtInp19", label = "Telephone :")),
    tags$div(id = "inline", textInput(inputId = "txtInp20", label = "Foreign worker:")),
    
    mainPanel(
      textOutput("selected_var")
  )
)
)


server <- function(input, output){
  
  output$selected_var <- renderText({ 
    
    Result <- lubridate:::predictRisk(input$txtInp1,input$txtInp12,input$txtInp3,input$txtInp4,
                input$txtInp4,input$txtInp5,input$txtInp6,input$txtInp7,
                input$txtInp8,input$txtInp9,input$txtInp10,input$txtInp11,
                input$txtInp12,input$txtInp13,input$txtInp14,input$txtInp15,
                input$txtInp16,input$txtInp17,input$txtInp18,input$txtInp19,
                input$txtInp20)
  
    paste("You have selected", Result)
  })
}


shinyApp(ui, server)
