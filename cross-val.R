unloadNamespace("readr")
library(readr)
library(plyr)
library(nnet)
library (MASS)
library (tree)
library (randomForest)
setwd("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump/")
wd_path <- getwd()
set.seed(1)

hads <- read_csv("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump/merged_data/small_dataset_file.csv", 
                 col_types = cols(label = col_factor(levels = c("sitting", 
                                                                "standing", "walking", "laying_down")), 
                                  timestamp = col_character()))

library(scales)

names(hads)
dim(hads)
summary(hads)
hads <- hads[order(hads$timestamp),]
drops <- c("rotation_vector_X6","rotation_vector_X7")
hads <- hads[ , !(names(hads) %in% drops)]
hads$label <- factor(hads$label, levels=sort(unique(hads$label)))
attach(hads)
View(hads)


train_rows <- sample(1:nrow(hads),0.9*nrow(hads)) #90% training and 10% test
test <- hads[-train_rows, 3:ncol(hads)]
training <- hads[train_rows, 3:ncol(hads)]

data <- training

k = 5 #Folds
# sample from 1 to k, nrow times (the number of observations in the data)
data$id <-sample(1:k, nrow(data), replace = TRUE)
list <- 1:k

model <- randomForest

# prediction and testset data frames that we add to with each iteration over
# the folds
prediction <- data.frame()
testsetCopy <- data.frame()
#Creating a progress bar to know the status of CV
progress.bar <- create_progress_bar("text")
progress.bar$init(k)
for (i in 1:k){
  
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  
  # run a model
  mymodel_ac <- model(label~(accelerometer_X2+accelerometer_X3+accelerometer_X4+accelerometer_X5),data=trainingset, na.action = na.omit, importance =TRUE)
  mymodel_gr <- model(label~(gravity_X2+gravity_X3+gravity_X4+gravity_X5),data=trainingset, na.action = na.omit, importance =TRUE)
  mymodel_gy <- model(label~(gyroscope_X2+gyroscope_X3+gyroscope_X4+gyroscope_X5),data=trainingset, na.action = na.omit, importance =TRUE)
  mymodel_la <- model(label~(linear_acceleration_X2+linear_acceleration_X3+linear_acceleration_X4+linear_acceleration_X5),data=trainingset, na.action = na.omit, importance =TRUE)
  mymodel_mf <- model(label~(magnetic_field_X2+magnetic_field_X3+magnetic_field_X4+magnetic_field_X5),data=trainingset, na.action = na.omit, importance =TRUE)
  mymodel_or <- model(label~(orientation_X2+orientation_X3+orientation_X4+orientation_X5),data=trainingset, na.action = na.omit, importance =TRUE)
  mymodel_rv <- model(label~(rotation_vector_X2+rotation_vector_X3+rotation_vector_X4+rotation_vector_X5),data=trainingset, na.action = na.omit, importance =TRUE)
  
  type <- "prob" #posterior or vector or prob
  predicted_scores_ac <- predict (mymodel_ac, testset, type)
  predicted_scores_gr <- predict (mymodel_gr, testset, type)
  predicted_scores_gy <- predict (mymodel_gy, testset, type)
  predicted_scores_la <- predict (mymodel_la, testset, type)
  predicted_scores_mf <- predict (mymodel_mf, testset, type)
  predicted_scores_or <- predict (mymodel_or, testset, type)
  predicted_scores_rv <- predict (mymodel_rv, testset, type)
  
  predicted_scores_ac[is.na(predicted_scores_ac)] <- 0
  predicted_scores_gr[is.na(predicted_scores_gr)] <- 0
  predicted_scores_gy[is.na(predicted_scores_gy)] <- 0
  predicted_scores_la[is.na(predicted_scores_la)] <- 0
  predicted_scores_mf[is.na(predicted_scores_mf)] <- 0
  predicted_scores_or[is.na(predicted_scores_or)] <- 0
  predicted_scores_rv[is.na(predicted_scores_rv)] <- 0
  
  predicted_scores <- predicted_scores_ac + predicted_scores_gr+predicted_scores_gy+predicted_scores_la+predicted_scores_mf+ predicted_scores_or+ predicted_scores_rv
  
  # make model predictions
  #temp <- as.data.frame(predict(mymodel, testset,type="prob"))
  
  predicted_scores_norm <- apply(predicted_scores, 2, rescale)
  predicted_labels <- apply(predicted_scores_norm, 1, which.max)
  predicted_labels <- as.factor(predicted_labels)
  levels(predicted_labels) <- c("sitting", "standing", "walking", "laying_down")
  
  temp <- predicted_labels
  
  # append this iteration's predictions to the end of the prediction data frame
  prediction <- rbind(prediction, as.data.frame(temp))
  
  
  # append this iteration's test set to the test set copy data frame
  # keep only the Sepal Length Column
  testsetCopy <- rbind(testsetCopy, as.data.frame(testset$label))
  
  progress.bar$step()
}
# add predictions and actual values
result <- cbind(prediction, testsetCopy)
names(result) <- c("Predicted", "Actual")

mean(as.character(result$Actual) == as.character(result$Predicted), na.rm = TRUE) #81.48 (multinom) and 81 with lda and 88 with qda and 0.78 with tree and 0.988 with random forest
