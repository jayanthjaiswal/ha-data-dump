unloadNamespace("readr")
library(readr)
merged_file <- read_csv("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump/merged_data_deb/merged_file.csv", 
                        col_types = cols(label = col_factor(levels = c("sitting", 
                                                                       "standing", "walking", "laying_down")), 
                                         timestamp = col_character()))
View(merged_file)
hads <- merged_file
hads <- merged_file[order(hads$timestamp),]
drops <- c("rotation_vector_X6","rotation_vector_X7")
hads <- hads[ , !(names(hads) %in% drops)]
hads$label <- factor(hads$label, levels=sort(unique(hads$label)))
attach(hads)
View(hads)

test <- hads[, 2:ncol(hads)]

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
