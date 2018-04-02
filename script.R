unloadNamespace("readr")
library(readr)
library(dplyr)
setwd("~/Courses/Winter/HealthAnalytics/HA-Project/ha-data-dump")
wd_path <- getwd()

folder_list <-
  dir(
    path = ".",
    pattern = "14442D1DF8A9310_Tue_Feb_27*",
    all.files = FALSE,
    full.names = FALSE,
    recursive = FALSE,
    ignore.case = FALSE
  )

output_folder <- "experiment"


for (folder in folder_list) {
  #folder <- "14442D1DF8A9310_Mon_Feb_19_15-48_2018_PST"
  prior_path <- paste(wd_path, folder, sep = "/")
  
  #X1 is timestamp in each file, last column (generally X6) is factor column storing the activity label
  # the summary stats based data selection is on sitting and walking
  
  linear_acceleration <-
    read_csv(
      paste(
        prior_path,
        "data/10_android.sensor.linear_acceleration.data.csv",
        sep = "/"
      ),
      col_names = FALSE,
      col_types = cols(X1 = col_character(),
                       X6 = col_factor(
                         levels = c("sitting", "standing", "walking", "laying_down")
                       ))
    )
  comment(linear_acceleration) <-
    deparse(substitute(linear_acceleration))
  colnames(linear_acceleration)[-1] <-
    paste(comment(linear_acceleration),
          colnames(linear_acceleration)[-1],
          sep = "_")
  # View(linear_acceleration)
  summary(linear_acceleration) # X1, X5, X6 - 221134 lines | all - 12986 lines
  
  
  rotation_vector <-
    read_csv(
      paste(
        prior_path,
        "data/11_android.sensor.rotation_vector.data.csv",
        sep = "/"
      ),
      col_names = FALSE,
      col_types = cols(X1 = col_character(),
                       X8 = col_factor(
                         levels = c("sitting", "standing", "walking", "laying_down")
                       ))
    )
  comment(rotation_vector) <- deparse(substitute(rotation_vector))
  colnames(rotation_vector)[-1] <-
    paste(comment(rotation_vector), colnames(rotation_vector)[-1], sep = "_")
  # View(rotation_vector)
  summary(rotation_vector) #all - 145779 lines
  
  step_counter <-
    read_csv(
      paste(
        prior_path,
        "data/19_android.sensor.step_counter.data.csv",
        sep = "/"
      ),
      col_names = FALSE,
      col_types = cols(X1 = col_character(),
                       X4 = col_factor(
                         levels = c("sitting", "standing", "walking", "laying_down")
                       ))
    )
  comment(step_counter) <- deparse(substitute(step_counter))
  colnames(step_counter)[-1] <-
    paste(comment(step_counter), colnames(step_counter)[-1], sep = "_")
  # View(step_counter)
  summary(step_counter) #not useful - 3 lines | all - 94 lines counts steps
  
  accelerometer <-
    read_csv(
      paste(
        prior_path,
        "data/1_android.sensor.accelerometer.data.csv",
        sep = "/"
      ),
      col_names = FALSE,
      col_types = cols(X1 = col_character(),
                       X6 = col_factor(
                         levels = c("sitting", "standing", "walking", "laying_down")
                       ))
    )
  comment(accelerometer) <- deparse(substitute(accelerometer))
  colnames(accelerometer)[-1] <-
    paste(comment(accelerometer), colnames(accelerometer)[-1], sep = "_")
  
  # l <- zoo(accelerometer[,2:5], order.by = accelerometer$X1)
  # lm <- rollapply(l, 5, mean, partial = TRUE)
  # colnames(lm) <- paste("lm5", colnames(accelerometer)[2:5], sep = "_")
  # lv <- rollapply(l, 5, var, partial = TRUE)
  # colnames(lv) <- paste("lv5", colnames(accelerometer)[2:5], sep = "_")
  # lk <- rollapply(l, 5, kurtosis, partial = TRUE)
  # colnames(lk) <- paste("lk5", colnames(accelerometer)[2:5], sep = "_")
  # ls <- rollapply(l, 5, skewness, partial = TRUE)
  # colnames(ls) <- paste("ls5", colnames(accelerometer)[2:5], sep = "_")
  # 
  # accelerometer <- cbind(accelerometer, as.data.frame(lm), as.data.frame(lv), as.data.frame(lk), as.data.frame(ls))
  # View(accelerometer)
  summary(accelerometer) #all - 221294 lines
  
  magnetic_field <-
    read_csv(
      paste(
        prior_path,
        "data/2_android.sensor.magnetic_field.data.csv",
        sep = "/"
      ),
      col_names = FALSE,
      col_types = cols(X1 = col_character(),
                       X6 = col_factor(
                         levels = c("sitting", "standing", "walking", "laying_down")
                       ))
    )
  comment(magnetic_field) <- deparse(substitute(magnetic_field))
  colnames(magnetic_field)[-1] <-
    paste(comment(magnetic_field), colnames(magnetic_field)[-1], sep = "_")
  # View(magnetic_field)
  summary(magnetic_field) #all - 75408 lines
  
  orientation <-
    read_csv(
      paste(
        prior_path,
        "data/3_android.sensor.orientation.data.csv",
        sep = "/"
      ),
      col_names = FALSE,
      col_types = cols(X1 = col_character(),
                       X6 = col_factor(
                         levels = c("sitting", "standing", "walking", "laying_down")
                       ))
    )
  comment(orientation) <- deparse(substitute(orientation))
  colnames(orientation)[-1] <-
    paste(comment(orientation), colnames(orientation)[-1], sep = "_")
  # View(orientation)
  summary(orientation) #all - 145779 lines
  
  gyroscope <-
    read_csv(
      paste(
        prior_path,
        "data/4_android.sensor.gyroscope.data.csv",
        sep = "/"
      ),
      col_names = FALSE,
      col_types = cols(X1 = col_character(),
                       X6 = col_factor(
                         levels = c("sitting", "standing", "walking", "laying_down")
                       ))
    )
  comment(gyroscope) <- deparse(substitute(gyroscope))
  colnames(gyroscope)[-1] <-
    paste(comment(gyroscope), colnames(gyroscope)[-1], sep = "_")
  # View(gyroscope)
  summary(gyroscope) #all - 145684 lines
  
  light <-
    read_csv(
      paste(prior_path, "data/5_android.sensor.light.data.csv", sep = "/"),
      col_names = FALSE,
      col_types = cols(X1 = col_character(),
                       X6 = col_factor(
                         levels = c("sitting", "standing", "walking", "laying_down")
                       ))
    )
  comment(light) <- deparse(substitute(light))
  colnames(light)[-1] <-
    paste(comment(light), colnames(light)[-1], sep = "_")
  # View(light)
  summary(light) #not useful - 38 lines
  
  # any_motion <-
  #   read_csv(
  #     paste(prior_path, "data/65539_.data.csv", sep = "/"),
  #     col_names = FALSE,
  #     col_types = cols(X1 = col_character(),
  #                      X18 = col_factor(
  #                        levels = c("sitting", "standing", "walking", "laying_down")
  #                      ))
  #   )
  # comment(any_motion) <- deparse(substitute(any_motion))
  # colnames(any_motion)[-1] <-
  #   paste(comment(any_motion), colnames(any_motion)[-1], sep = "_")
  # # View(any_motion)
  # summary(any_motion) #not useful - 73 lines
  
  gravity <-
    read_csv(
      paste(prior_path, "data/9_android.sensor.gravity.data.csv", sep = "/"),
      col_names = FALSE,
      col_types = cols(X1 = col_character(),
                       X6 = col_factor(
                         levels = c("sitting", "standing", "walking", "laying_down")
                       ))
    )
  comment(gravity) <- deparse(substitute(gravity))
  colnames(gravity)[-1] <-
    paste(comment(gravity), colnames(gravity)[-1], sep = "_")
  # View(gravity)
  summary(gravity) #X1 X5 X6 - 221134 lines | all - 12986 lines
  
  #not using any_motion, light and step_counter - step_counter is also seperating feature
  merged_data <-
    list(
      accelerometer,
      gravity,
      gyroscope,
      linear_acceleration,
      magnetic_field,
      orientation,
      rotation_vector
    ) %>%
    Reduce(function(dtf1, dtf2)
      full_join(dtf1, dtf2, by = "X1"), .)
  
  output_location <-
    paste(output_folder , paste(folder, "csv", sep = "."), sep = "/")
  
  merged_data <- within(
    merged_data,
    label <- list(
      accelerometer_X6,
      gravity_X6,
      gyroscope_X6,
      linear_acceleration_X6,
      magnetic_field_X6,
      orientation_X6,
      rotation_vector_X8
    ) %>%
      Reduce(function(A, B)
        ifelse(
          !is.na(A), as.character(A), as.character(B)
        ), .)
  )
  
  drops <- c(
    "accelerometer_X6",
    "gravity_X6",
    "gyroscope_X6",
    "linear_acceleration_X6",
    "magnetic_field_X6",
    "orientation_X6",
    "rotation_vector_X8"
  )
  
  merged_data <- merged_data[, !(names(merged_data) %in% drops)]
  colnames(merged_data)[1] <- "timestamp"
  
  write.csv(merged_data, file = output_location, row.names=FALSE)
  cat("folder done - ", folder)
  flush.console()
}

multmerge = function(mypath){
  filenames=list.files(path=mypath, full.names=TRUE)
  datalist = lapply(filenames, function(x){read.csv(file=x,header=T)})
  Reduce(function(x,y) {rbind(x,y)}, datalist)
}

merged_data = multmerge(output_folder)
output_location <-
  paste(output_folder , paste("merged_file", "csv", sep = "."), sep = "/")

write.csv(merged_data, file = output_location, row.names=FALSE)

#creating small dataset
total_size <- 1000000
small_dataset <- sample_n(merged_data, total_size)
summary(small_dataset)
output_location <-
paste(output_folder , paste("small_dataset_file", "csv", sep = "."), sep = "/")
write.csv(small_dataset, file = output_location, row.names=FALSE)

