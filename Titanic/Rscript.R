setwd("~/Documents/PROJECTS/contests/Titanic/")
source("~/Documents/WORK/source_functions.R")

library(ggplot2)
# library(e1071)
library(randomForest)
library(rpart)
library(Hmisc)
library(nnet)

df.train <- read.csv("DATA/train.csv", na.strings="NA", header=TRUE)
df.train$survived <- as.factor(df.train$survived)
df.test <- read.csv("DATA/test.csv", na.strings="NA", header=TRUE)

############################################################################
## GLM
############################################################################
m <- glm(survived ~ pclass+age+sibsp+parch+sex, df.train, family="binomial")
df.predict <- predict(m, df.test, type="response")
# df.predict <- round(df.predict, digits=0)
df.predict <- ifelse(is.na(df.predict), 0, df.predict)
# write.csv(df.predict, file="predictions.csv", row.names=FALSE, col.names=NA)

############################################################################
## Tree Based Model
############################################################################
df.imputed <- rfImpute(survived ~ pclass+age+parch+sex+embarked, df.train)
m.tree <- rpart(survived ~ ., df.imputed)
df.predict <- predict(m.tree, df.test, type="class")
df.predict <- as.integer(df.predict) - 1
# write.csv(df.predict, file="predictions_tree.csv", row.names=FALSE)
table(df.predict)


############################################################################
## Neural Network
############################################################################
x.nn <- subset(df.imputed, select = c("pclass","age","parch","sex","embarked"))
y.nn <- subset(df.imputed, select = "survived")
m.nn <- nnet(survived ~ pclass+age+parch+sex+embarked, df.imputed, size=4)

df.test$age <- impute(df.test$age, fun=median)
df.predict <- predict(m.nn, df.test, type="raw")
df.predict <- round(df.predict)
table(df.predict)
write.csv(df.predict, file="predictions_tree.csv", row.names=FALSE)



