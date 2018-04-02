library(readr)
library(plyr)
library(nnet)
library (MASS)
library (tree)
library (randomForest)
setwd("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump/")
wd_path <- getwd()

hads <- read_csv("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump/merged_data/small_dataset_file.csv", 
                 col_types = cols(label = col_factor(levels = c("sitting", 
                                                                "standing", "walking", "laying_down")), 
                                  timestamp = col_character()))
names(hads)
dim(hads)
summary(hads)
hads <- hads[order(hads$timestamp),]
drops <- c("rotation_vector_X6","rotation_vector_X7")
hads <- hads[ , !(names(hads) %in% drops)]
hads$label <- factor(hads$label, levels=sort(unique(hads$label)))
attach(hads)
View(hads)

set.seed(1)

train_rows <- sample(1:nrow(hads),0.9*nrow(hads)) #90% training and 10% test
test <- hads[-train_rows, 3:ncol(hads)]
training <- hads[train_rows, 3:ncol(hads)]

###
model <- randomForest
# mymodel_ac <- model(label~(accelerometer_X2+accelerometer_X3+accelerometer_X4+accelerometer_X5),data=training, family="binomial", maxit=1000)
# mymodel_gr <- model(label~(gravity_X2+gravity_X3+gravity_X4+gravity_X5),data=training, family="binomial", maxit=1000)
# mymodel_gy <- model(label~(gyroscope_X2+gyroscope_X3+gyroscope_X4+gyroscope_X5),data=training, family="binomial", maxit=1000)
# mymodel_la <- model(label~(linear_acceleration_X2+linear_acceleration_X3+linear_acceleration_X4+linear_acceleration_X5),data=training, family="binomial", maxit=1000)
# mymodel_mf <- model(label~(magnetic_field_X2+magnetic_field_X3+magnetic_field_X4+magnetic_field_X5),data=training, family="binomial", maxit=1000)
# mymodel_or <- model(label~(orientation_X2+orientation_X3+orientation_X4+orientation_X5),data=training, family="binomial", maxit=1000)
# mymodel_rv <- model(label~(rotation_vector_X2+rotation_vector_X3+rotation_vector_X4+rotation_vector_X5),data=training, family="binomial", maxit=1000)

mymodel_ac <- model(label~(accelerometer_X2+accelerometer_X3+accelerometer_X4+accelerometer_X5),data=training, na.action = na.omit, importance =TRUE)
mymodel_gr <- model(label~(gravity_X2+gravity_X3+gravity_X4+gravity_X5),data=training, na.action = na.omit, importance =TRUE)
mymodel_gy <- model(label~(gyroscope_X2+gyroscope_X3+gyroscope_X4+gyroscope_X5),data=training, na.action = na.omit, importance =TRUE)
mymodel_la <- model(label~(linear_acceleration_X2+linear_acceleration_X3+linear_acceleration_X4+linear_acceleration_X5),data=training, na.action = na.omit, importance =TRUE)
mymodel_mf <- model(label~(magnetic_field_X2+magnetic_field_X3+magnetic_field_X4+magnetic_field_X5),data=training, na.action = na.omit, importance =TRUE)
mymodel_or <- model(label~(orientation_X2+orientation_X3+orientation_X4+orientation_X5),data=training, na.action = na.omit, importance =TRUE)
mymodel_rv <- model(label~(rotation_vector_X2+rotation_vector_X3+rotation_vector_X4+rotation_vector_X5),data=training, na.action = na.omit, importance =TRUE)


type <- "prob" #posterior or vector or prob
predicted_scores_ac <- predict (mymodel_ac, test, type)
predicted_scores_gr <- predict (mymodel_gr, test, type)
predicted_scores_gy <- predict (mymodel_gy, test, type)
predicted_scores_la <- predict (mymodel_la, test, type)
predicted_scores_mf <- predict (mymodel_mf, test, type)
predicted_scores_or <- predict (mymodel_or, test, type)
predicted_scores_rv <- predict (mymodel_rv, test, type)

predicted_scores_ac[is.na(predicted_scores_ac)] <- 0
predicted_scores_gr[is.na(predicted_scores_gr)] <- 0
predicted_scores_gy[is.na(predicted_scores_gy)] <- 0
predicted_scores_la[is.na(predicted_scores_la)] <- 0
predicted_scores_mf[is.na(predicted_scores_mf)] <- 0
predicted_scores_or[is.na(predicted_scores_or)] <- 0
predicted_scores_rv[is.na(predicted_scores_rv)] <- 0

predicted_scores <- predicted_scores_ac + predicted_scores_gr+predicted_scores_gy+predicted_scores_la+predicted_scores_mf+ predicted_scores_or+ predicted_scores_rv
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

#0.922 & 0.80 with acc(1183) & 0.78 with gravity(2126) & 0.67 with gyro(2055) & 
#0.68 with la(2126) & 0.75 with magnetic field(5976) & 0.78 with orient (2049) & 0.82 with rotation_vector (2049)

#0.70 with tree and no NAs
#0.99 with Random Forest with 6920 NA

table(predicted_labels, test$label)
mean(predicted_labels == test$label, na.rm = TRUE) #0.9025974 & 0.78 with acc 

importance (mymodel_ac)
importance (mymodel_gr)
importance (mymodel_gy)
importance (mymodel_la)
importance (mymodel_mf)
importance (mymodel_or)
importance (mymodel_rv)

varImpPlot (mymodel_ac)
varImpPlot (mymodel_gr)
varImpPlot (mymodel_gy)
varImpPlot (mymodel_la)
varImpPlot (mymodel_mf)
varImpPlot (mymodel_or)
varImpPlot (mymodel_rv)


