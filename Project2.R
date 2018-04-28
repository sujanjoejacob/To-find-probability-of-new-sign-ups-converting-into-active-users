#install.packages("e1071")
#install.packages("ElemStatLearn")
library(reshape2)
library(ggplot2)
library(pander)
library(pROC)
library(h2o)
#install.packages("lsr")
#install.packages('reshape2')
#install.packages("ROCR")
library(caret)
library(knitr)
library(ElemStatLearn)
library(gbm)
#install.packages("GGally")
#install.packages("gbm")
library(data.table)
library(testthat)
library(gridExtra)
library(corrplot)
library(GGally)
library(ggplot2)
library(e1071)
library(dplyr)
#install.packages("pastecs")
#install.packages("cvAUC")
library(xgboost)
library(Matrix)
library(cvAUC)


library(Hmisc)
library(pastecs)
library(psych)


field<- read.csv("Fieldsense_2.csv")
View(field)

colnames(field)
dim(field)
str(field)
summary(field)

#Finding missing values
table(field$Designation)
str(field)

#Check NAs and less than 0 values
sapply(field, function(x) sum(is.na(x)))

#Univariate analysis

#Chi q. test 
table(field$CustomerType, field$UserStatus)

#for Categorical & Categorical:Pearson's Chi-squared test

chisq.test(field$CustomerType, field$UserStatus, correct=FALSE)
#We have a chi-squared value of 13.002 and p value = 0.0003112 Since we get a p-Value less than the significance level of 0.05, we reject the null hypothesis and conclude that the two variables 

#DOMAIN
table(field$Domain, field$UserStatus)
chisq.test(field$Domain, field$UserStatus, correct = FALSE) 

#type
table(field$Type, field$UserStatus)
chisq.test(field$Type, field$UserStatus, correct = FALSE) 

#Enq. freq
table(field$Enquiry_freq, field$UserStatus)
chisq.test(field$Enquiry_freq, field$UserStatus, correct = FALSE)

#ANOVA
field$UserStatus <- as.factor(field$UserStatus)
aov1 = aov(field$Industry ~ field$UserStatus)
summary(aov1)

stat.desc(field$Size)

boxplot(field$Year,
        main = toupper("Boxplot of Year of Estb."),
        ylab = "Year",
        col = "blue")

#Kernal desity plot 
d <- density(field$Year)
plot(d, main = "Kernel density of year of Estb.")
polygon(d, col = "red", border = "blue")

#Kernal desity plot 
d <- density(field$Size)
plot(d, main = "Kernel density of Size")
polygon(d, col = "red", border = "blue")

#for year
field$Year_1 <- as.numeric(field$Year >= 41 & field$Year <= 1000)
field$Year_2 <- as.numeric(field$Year >= 20 & field$Year <= 40)
field$Year_3 <- as.numeric(field$Year >= 10 & field$Year <= 19)
field$Year_4 <- as.numeric(field$Year >= 6 & field$Year <= 9)
field$Year_5 <- as.numeric(field$Year >= 1 & field$Year <= 5)

field$Year <- NULL

#for Size
field$Size[field$Size>=500 & field$Size<10000]=10000
field$Size[field$Size>=100 & field$Size<200]=200
field$Size[field$Size>=10 & field$Size<75]=75


#feature Extraction
field$Contacted <- NULL
field$Contact.Person <- NULL
field$Tel.No <- NULL
field$Email.ID <- NULL
field$Invoicee <- NULL
field$Domain <-NULL

View(field)

#field <- field[,c(11,1,2,3,5:11)] #Reorder variables to put target variable to the first place
#field$UserStatus.1 <- NULL

# dummy variables for factors/characters
field$CustomerType <- as.factor(field$CustomerType)
fielddummy <- dummyVars("~.",data=field, fullRank=F)
field_1 <- as.data.frame(predict(fielddummy,field))
print(names(field))


View(field_1)
str(field_1)
str(UserStatus)


#field_1$UserStatus.0 <- NULL
#field_1$UserStatus.1 <- NULL

#added usestatus column
field_1$UserStatus <- paste(field$UserStatus)


View(field_1)
str(field_1)

# Encoding the target feature as factor
field_1$UserStatus <- as.factor(field_1$UserStatus)

# what is the proportion of your outcome variable?
prop.table(table(field_1$UserStatus))

 # save the outcome for the glmnet model
tempOutcome <- field_1$UserStatus

# generalize outcome and predictor variables
outcomeName <- 'UserStatus'
predictorsNames <- names(field_1)[names(field_1) != outcomeName]

# model it
#################################################
# get names of all caret supported models 
names(getModelInfo())


field_1$UserStatus <- ifelse(field_1$UserStatus==1,'Active','Inactive')

# pick model gbm and find out what type of model it is
getModelInfo()$gbm$type

# split data into training and testing chunks
set.seed(1234)

splitIndex <- createDataPartition(field_1[,"UserStatus"], p = .75, list = FALSE, times = 1)
trainDF <- field_1[ splitIndex,]
testDF  <- field_1[-splitIndex,]

View(testDF)
View(trainDF)
dim(trainDF)
dim(testDF)

splitIndex

# create caret trainControl object to control the number of cross-validations performed
objControl <- trainControl(method='cv', number=5, returnResamp='none',classProbs = TRUE)


# run model
objModel <- train(trainDF[,predictorsNames], as.factor(trainDF[,outcomeName]), 
                  method='gbm', 
                  trControl=objControl,  
                  metric = "ROC",
                  preProc = c("center", "scale"))


# find out variable importance
summary(objModel)

# find out model details
objModel


# evalutate model
#################################################
# get predictions on your testing data

# class prediction
predictions <- predict(object=objModel, testDF[,predictorsNames], type='raw')
head(predictions)
postResample(pred=predictions, obs=as.factor(testDF[,outcomeName]))


varImpPlot(objModel)


library(klaR)
#confusion matrix
prdval <- predict(objModel, testDF)
table(testDF$UserStatus, prdval)
testDF


# probabilities 
predictions <- predict(object=objModel, testDF[,predictorsNames], type='prob')
View(predictions)
postResample(pred=predictions[[2]], obs=ifelse(testDF[,outcomeName]=='yes',1,0))

#PRINT roc
auc <- roc(ifelse(testDF[,outcomeName]=="Active",1,0), predictions[[2]])
print(auc$auc)

plot(varImp(objModel,scale=F))


#Exporting the Prob. on Test data set
write.csv(predictions, "Probabilty_test.csv")
write.csv(testDF, "Test_test.csv")

View(field)

library(ggplot2)
ggplot(field, aes(x=Size, y=Year)) + geom_point()

################################################
# glmnet model
################################################

# pick model gbm and find out what type of model it is
getModelInfo()$glmnet$type 


# save the outcome for the glmnet model
field_1$UserStatus  <- tempOutcome
field_1$UserStatus <- as.factor(field_1$UserStatus)
str(field_1)
View(field_1$UserStatus)

# split data into training and testing chunks
set.seed(1234)
splitIndex <- createDataPartition(field_1[,outcomeName], p = .75, list = FALSE, times = 1)

trainDF <- field_1[ splitIndex,]
testDF  <- field_1[-splitIndex,]


# create caret trainControl object to control the number of cross-validations performed
objControl <- trainControl(method='cv', number=3, returnResamp='none')


# run model
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], method='glmnet',  metric = "Roc", trControl=objControl)


# get predictions on your testing data
predictions <- predict(object=objModel, testDF[,predictorsNames])

library(pROC)
auc <- roc(testDF[,outcomeName], predictions)
print(auc$auc)

# find out variable importance
summary(objModel)
plot(varImp(objModel,scale=F))

# find out model details
objModel

# display variable importance on a +/- scale 
vimp <- varImp(objModel, scale=F)
results <- data.frame(row.names(vimp$importance),vimp$importance$Overall)
results$VariableName <- rownames(vimp)
colnames(results) <- c('VariableName','Weight')
results <- results[order(results$Weight),]
results <- results[(results$Weight != 0),]


par(mar=c(5,15,4,2)) # increase y-axis margin. 
xx <- barplot(results$Weight, width = 0.85, 
              main = paste("Variable Importance -",outcomeName), horiz = T, 
              xlab = "< (-) importance >  < neutral >  < importance (+) >", axes = FALSE, 
              col = ifelse((results$Weight > 0), 'blue', 'red')) 
axis(2, at=xx, labels=results$VariableName, tick=FALSE, las=2, line=-0.3, cex.axis=0.6)  


################################################
# advanced stuff
################################################

# boosted tree model (gbm) adjust learning rate and and trees
gbmGrid <-  expand.grid(interaction.depth =  c(1, 5, 9),
                        n.trees = 50,
                        shrinkage = 0.01)
# run model
objModel <- train(trainDF[,predictorsNames], trainDF[,outcomeName], method='gbm', trControl=objControl, tuneGrid = gbmGrid, verbose=F)


# get predictions on your testing data
predictions <- predict(object=objModel, testDF[,predictorsNames])

View(testDF)

plot(CustomerType~UserStatus,field)

plot(Industry~UserStatus,field)

View(predictions)

boxplot(formula=as.numeric(field$".Phase")~data$"year",col="blue")

