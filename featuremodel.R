library(readr)
library(plyr)
library(nnet)
library (MASS)
library (tree)
library (randomForest)
library(dplyr)
setwd("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump/")
wd_path <- getwd()

hads <- read_csv("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump/merged_jay.csv", 
                 col_names = FALSE)

names(hads) <- c("timestamp","accelerometer_X2","accelerometer_X3","accelerometer_X4","accelerometer_X5","label")
hads$label <- factor(hads$label, levels = c("sitting", 
                                            "standing", "walking", "laying_down"))
dim(hads)
summary(hads)
View(hads)

total_size <- 10000
small_dataset <- sample_n(hads, total_size)
summary(small_dataset)
hads<-small_dataset

attach(hads)

datasize <- dim(hads)[1]

hads$accelerometer_X2_EMA = numeric(datasize)
hads$accelerometer_X3_EMA = numeric(datasize)
hads$accelerometer_X4_EMA = numeric(datasize)

hads$accelerometer_X2_EMV = numeric(datasize)
hads$accelerometer_X3_EMV = numeric(datasize)
hads$accelerometer_X4_EMV = numeric(datasize)

rho <- 0.5 # N =9 is window contains 86% weight

hads$accelerometer_X2_EMA[1] <- (hads$accelerometer_X2[1])
hads$accelerometer_X3_EMA[1] <- (hads$accelerometer_X3[1])
hads$accelerometer_X4_EMA[1] <- (hads$accelerometer_X4[1])

progress.bar <- create_progress_bar("text")
progress.bar$init(datasize)

for (i in 2:datasize) {
  hads$accelerometer_X2_EMA[i] <- (hads$accelerometer_X2[i] - hads$accelerometer_X2_EMA[i-1])
  hads$accelerometer_X3_EMA[i] <- (hads$accelerometer_X3[i] - hads$accelerometer_X3_EMA[i-1])
  hads$accelerometer_X4_EMA[i] <- (hads$accelerometer_X4[i] - hads$accelerometer_X4_EMA[i-1])
  
  hads$accelerometer_X2_EMV[i] = (1 - rho)*(hads$accelerometer_X2_EMV[i-1] + rho * (hads$accelerometer_X2_EMA[i-1]**2))
  hads$accelerometer_X3_EMV[i] = (1 - rho)*(hads$accelerometer_X3_EMV[i-1] + rho * (hads$accelerometer_X3_EMA[i-1]**2))
  hads$accelerometer_X4_EMV[i] = (1 - rho)*(hads$accelerometer_X4_EMV[i-1] + rho * (hads$accelerometer_X4_EMA[i-1]**2))
  
  hads$accelerometer_X2_EMA[i] = hads$accelerometer_X2_EMA[i-1] + rho*hads$accelerometer_X2_EMA[i]
  hads$accelerometer_X3_EMA[i] = hads$accelerometer_X3_EMA[i-1] + rho*hads$accelerometer_X3_EMA[i]
  hads$accelerometer_X4_EMA[i] = hads$accelerometer_X4_EMA[i-1] + rho*hads$accelerometer_X4_EMA[i]
  progress.bar$step()
}

set.seed(1)

train_rows <- sample(1:nrow(hads),0.9*nrow(hads)) #90% training and 10% test
test <- hads[-train_rows, 2:ncol(hads)]
training <- hads[train_rows, 2:ncol(hads)]

###
model <- randomForest
#mymodel_ac <- model(label~(accelerometer_X2+accelerometer_X3+accelerometer_X4+accelerometer_X5),data=training)
# mymodel_gr <- model(label~(gravity_X2+gravity_X3+gravity_X4+gravity_X5),data=training, family="binomial", maxit=1000)
# mymodel_gy <- model(label~(gyroscope_X2+gyroscope_X3+gyroscope_X4+gyroscope_X5),data=training, family="binomial", maxit=1000)
# mymodel_la <- model(label~(linear_acceleration_X2+linear_acceleration_X3+linear_acceleration_X4+linear_acceleration_X5),data=training, family="binomial", maxit=1000)
# mymodel_mf <- model(label~(magnetic_field_X2+magnetic_field_X3+magnetic_field_X4+magnetic_field_X5),data=training, family="binomial", maxit=1000)
# mymodel_or <- model(label~(orientation_X2+orientation_X3+orientation_X4+orientation_X5),data=training, family="binomial", maxit=1000)
# mymodel_rv <- model(label~(rotation_vector_X2+rotation_vector_X3+rotation_vector_X4+rotation_vector_X5),data=training, family="binomial", maxit=1000)

mymodel_ac <- model(label~(accelerometer_X2+accelerometer_X3+accelerometer_X4+accelerometer_X5+accelerometer_X2_EMA+accelerometer_X3_EMA+accelerometer_X4_EMA+accelerometer_X2_EMV+accelerometer_X3_EMV+accelerometer_X4_EMV),data=training, na.action = na.omit)

type <- "prob" #posterior or vector or prob
predicted_scores_ac <- predict (mymodel_ac, test, type)

predicted_scores_ac[is.na(predicted_scores_ac)] <- 0


predicted_scores <- predicted_scores_ac
library(scales)
predicted_scores_norm <- apply(predicted_scores, 2, rescale)
predicted_labels <- apply(predicted_scores_norm, 1, which.max)
predicted_labels <- as.factor(predicted_labels)
levels(predicted_labels) <- c("sitting", "standing", "walking", "laying_down")

#mymodel <- lda(label~.,data=training)
#mymodel <- qda(label~.,data=training)
#mymodel <- tree(label~.,data=training)
#mymodel <- randomForest(label~.,data=training , mtry=15, importance =TRUE, na.action = na.omit)
#mymodel <- gbm(label~.,data=training , distribution= "multinomial",n.trees =5000 , interaction.depth =4)

#summary(mymodel)
#head(pp <- fitted(mymodel))
#predicted_scores <- predict (mymodel, test, "probs")
#predicted_labels <- predict (mymodel, test)
#summary(predicted_labels)

table(predicted_labels, test$label)
mean(as.character(predicted_labels) == as.character(test$label), na.rm = TRUE) #81.48 (multinom) and 81 with lda and 88 with qda and 0.78 with tree and 0.988 with random forest
