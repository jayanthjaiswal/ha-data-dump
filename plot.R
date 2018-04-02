library(readr)
library(plyr)
setwd("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump/")
wd_path <- getwd()

hads <- read_csv("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump/merged_data/small_dataset_file.csv", 
                 col_types = cols(label = col_factor(levels = c("sitting", 
                                                                "standing", "walking", "laying_down")), 
                                  timestamp = col_character()))
View(hads)
names(hads)
dim(hads)
hads <- hads[order(timestamp),]
attach(hads)
cor(hads[, 3:32], use = "complete.obs")

plot.ts(hads[, 3:6])
plot.ts(hads[, 7:10])
plot.ts(hads[, 11:14])
plot.ts(hads[, 15:18])
plot.ts(hads[, 19:22])
plot.ts(hads[, 23:26])
plot.ts(hads[, 27:32])
plot.ts(hads[, 33])

pairs(hads[, 3:6])
pairs(hads[, 7:10])
pairs(hads[, 11:14])
pairs(hads[, 15:18])
pairs(hads[, 19:22])
pairs(hads[, 23:26])
pairs(hads[, 27:32])