Data source - https://www.kaggle.com/avikasliwal/used-cars-price-prediction

The best model was decided based on mape on the test data. The model that gave the least mape on the test data was the best model.

# Clearing the R environment
rm(list = ls( ))

# Importing all the required libraries
library(DataExplorer)
library(stringr)
library(dplyr)
library(DMwR)
library(ggplot2)
library(modeest)
library(corrplot)
library(randomForest)
library(xgboost)
library(readr)
library(caret)
library(tidyverse)

# Importing the given train data set
car<-read.csv(choose.files(),header=TRUE)
View(car)

# EXPLORATORY DATA ANALYSIS - EXPLORING THE RAW DATA SET

# Data dimension
dim(car)

# Viewing the first 10 observations of the data set
View(head(car,10))

# Variable names in the data set
names(car)

# Checking the structure to know the data type of each variable in the data set
str(car)

# Due to the presence of units, R is not able to identify all missing values. Hence we replaced all blanks with the word 'missing' in the csv file and then imported it.
# Replacing the word 'missing' with NA
car[car=='missing']<-NA

# Checking for missing values in the data set
plot_intro(car)
sum(is.na(car))

# Checking column wise % of missing values
colSums(is.na(car))/nrow(car)*100



# PREPROCESSING THE RAW DATA SET

# Separating units from numeric variables since units are not required for modelling and will be used only for interpretation
car$Engine <- gsub(" CC","",car$Engine)
car$Engine <- as.integer(car$Engine)
car$Power <- gsub(" bhp","",car$Power)
car$Power <- as.integer(car$Power)

# Checking Mileage by fuel type
car %>% pull(Mileage) %>% str_split(" ", simplify=TRUE) %>% cbind(car$Fuel_Type) %>% subset(select=2:3) %>% as.data.frame() %>% table()

# Since the two units under Mileage represent two different fuel types (km/kg for CNG or LPG cars) and (kmpl for Petrol or Diesel cars) we Split 'Mileage' into two variables 'kmpkg' and 'kmpl' and eliminated the variable 'Mileage'
car %>%mutate(Mileage = as.numeric(str_extract(Mileage, "^[:graph:]+"))) %>%mutate(kmpkg = ifelse(Fuel_Type=="CNG" | Fuel_Type=="LPG", Mileage, 0)) %>%mutate(kmpL = ifelse(Fuel_Type=="Diesel" | Fuel_Type=="Petrol", Mileage, 0)) %>%select(-Mileage) -> car
head(car)

# Eliminating the variable 'New_Price' since it has 86% (>30%) of missing values
car$New_Price<-NULL

# Re-checking for missing values
colSums(is.na(car))

# Imputing the remaining missing values under 'Engine', 'Power' and 'Seats' using knn imputation 

# Converting all character variables to factor for imputation
car[] <- lapply(car, function(x) if(is.character(x)) as.factor(x) else x)
car$Seats<-as.factor(car$Seats)
str(car)

# Knn imputation using 3 nearest neighbours
data_imputed = knnImputation(car[-12],k=3)

# Rechecking for missing values
sum(is.na(data_imputed))
names(data_imputed)

# Adding the excluded dependent variable to the data set
data_imputed$Price<-car$Price

# Creating a new variable 'Age_of_car' from 'Year'
data_imputed$Age_of_car <- as.integer(2021-car$Year)



# EXPLORATORY DATA ANALYSIS - EXPLORING THE PREPROCESSED DATA

# Data dimension
dim(data_imputed)

# Checking for duplicate records in the data set
sum(duplicated(data_imputed$ID))

# Eliminating the 39 records with '0 Seats' as a car cannot have 0 seats
data_imputed<-subset(data_imputed,Seats!="0")
dim(data_imputed)

# Creating a data frame of only the numeric/continuous variables
data_imputed_num<-data_imputed[,c(5,9,10,12,13,14,15)]

# Separating the Car company name and model name
y <- as.data.frame(str_split_fixed(data_imputed$Name, " ", 2))
colnames(y) <- c("Brand", "CarModel")
data_imputed<-cbind(data_imputed,y)
View(data_imputed)

# Summary of numeric/continuous variables after imputation
summary(data_imputed_num)

# Summary of categorical variables after imputation
data.frame(table(data_imputed$Seats))
data.frame(table(data_imputed$Location))
data.frame(table(data_imputed$Fuel_Type))
data.frame(table(data_imputed$Transmission))
data.frame(table(data_imputed$Owner_Type))
data.frame(table(data_imputed$Seats))
data.frame(table(data_imputed$Brand))



# DATA VISUALIZATION

# Distribution of all numeric/continuous variables in the data set
plot_histogram(data_imputed_num)

# Number of cars by location
ggplot(data_imputed, aes(Location)) +geom_bar(fill="maroon")+ggtitle("Number of cars by Location")

# Number of cars by Brand
ggplot(data_imputed, aes(Brand)) +geom_bar(fill="maroon")+ggtitle("Number of cars by Brand")+coord_flip()

# Number of cars by Fuel type
ggplot(data_imputed, aes(Fuel_Type)) +geom_bar(fill="maroon")+ggtitle("Number of cars by Fuel Type")

# Number of cars by Transmission
ggplot(data_imputed, aes(Transmission)) +geom_bar(fill="maroon")+ggtitle("Number of cars by Transmission")

# Number of cars by Owner type
ggplot(data_imputed, aes(Owner_Type)) +geom_bar(fill="maroon")+ggtitle("Number of cars by Owner Type")

# Number of cars by Seats
ggplot(data_imputed, aes(Seats)) +geom_bar(fill="maroon")+ggtitle("Number of cars by Seats")

# Which is the priciest car?

# Subsetting the data set by maximum price
max(data_imputed$Price)
priciest<- subset(data_imputed, Price == 160)
priciest[,c(-1,-4)]

# Which is the Cheapest car?

# Subsetting the data set by minimum price
min(data_imputed$Price)
cheapest<- subset(data_imputed, Price == 0.44)
cheapest[,c(-1,-4)]

# Which is the most sold car?
mfv(data_imputed$Name)

# Creating two separate data frames for Manual and Automatic cars
Manual <- subset(data_imputed, Transmission == "Manual")
Automatic<-subset(data_imputed, Transmission == "Automatic")

dim(Manual)
dim(Automatic)

# Finding the price range and average price for Manual cars
summary(Manual$Price)

# Finding the price range and average price for Automatic cars
summary(Automatic$Price)

# Which is the priciest Manual car?

# Subsetting the data set by maximum price
max(Manual$Price)
priciest<- subset(Manual, Price == 40)
priciest[,c(-1,-4)]

# Which is the Cheapest Manual car?

# Subsetting the data set by minimum price
min(Manual$Price)
cheapest<- subset(Manual, Price == 0.44)
cheapest[,c(-1,-4)]

# Which is the most sold Manual car?
mfv(Manual$Name)


# Which is the priciest Automatic  car?

# Subsetting the data set by maximum price
max(Automatic$Price)
priciest<- subset(Automatic, Price == 160)
priciest[,c(-1,-4)]

# Which is the Cheapest Automatic car?

# Subsetting the data set by minimum price
min(Automatic$Price)
cheapest<- subset(Automatic, Price == 1.5)
cheapest[,c(-1,-4)]

# Which is the most sold Automatic car?
mfv(Automatic$Name)

# Creating two separate data frames for Diesel/Petrol and CNG/LPG cars
LP_CN <- subset(data_imputed, Fuel_Type == "CNG" | Fuel_Type=="LPG")
PE_DI<- subset(data_imputed, Fuel_Type == "Petrol" | Fuel_Type=="Diesel")

dim(LP_CN)
dim(PE_DI)

# Finding the price range and average price for Diesel/Petrol
summary(PE_DI$Price)

# Finding the price range and average price for CNG/LPG cars
summary(LP_CN$Price)

# Which is the priciest Petrol/Diesel car?

# Subsetting the data set by maximum price
max(PE_DI$Price)
priciest<- subset(PE_DI, Price == 160)
priciest[,c(-1,-4)]

# Which is the Cheapest Petrol/Diesel car?

# Subsetting the data set by minimum price
min(PE_DI$Price)
cheapest<- subset(PE_DI, Price == 0.44)
cheapest[,c(-1,-4)]

# Which is the most sold/bought Petrol/Diesel car?
mfv(PE_DI$Name)

# Which is the priciest CNG/LPG car?

# Subsetting the data set by maximum price
max(LP_CN$Price)
priciest<- subset(LP_CN, Price == 8.35)
priciest[,c(-1,-4)]

# Which is the Cheapest CNG/LPG car?

# Subsetting the data set by minimum price
min(LP_CN$Price)
cheapest<- subset(LP_CN, Price == 1.2)
cheapest[,c(-1,-4)]

# Which is the most sold/bought CNG/LPG car?
mfv(LP_CN$Name)


# Correlation among the continuous variables

# Finding the correlation
corr<-cor(data_imputed_num)

# Correlation plot
corrplot(corr, method="color",addCoef.col = "black")

# Studying the relationship of all categorical variables with 'Price' using boxplots

# Price v/s Location
data_imputed %>% ggplot(aes(Location, Price,color=Location))+geom_boxplot()+geom_jitter(alpha=0.1)+scale_y_log10()+xlab('Location') +ylab('Price of the car') +ggtitle('Price vs. Location')

# Price v/s Age_of_data_imputed
data_imputed %>% ggplot(aes(Age_of_car, Price,color=Age_of_car))+geom_boxplot(aes(group=Age_of_car))+geom_jitter(alpha=0.1)+geom_smooth(method="loess")+scale_y_log10()+xlab('Age_of_car') +ylab('Price of the car') +ggtitle('Price vs. Age_of_car')

# Price v/s Fuel Type
ggplot(data = data_imputed, aes(x = Fuel_Type, y = Price,color=Fuel_Type)) +geom_boxplot() +xlab('Fuel_Type') +ylab('Price of the car') +ggtitle('Price vs. Fuel_type')

# Price v/s Transmission
data_imputed %>% ggplot(aes(Transmission, Price,color=Transmission))+geom_boxplot()+geom_jitter(alpha=0.1)+scale_y_log10()+xlab('Tansmission') +ylab('Price of the car') +ggtitle('Price vs. Transmission')

# Price v/s Owner type
data_imputed %>% ggplot(aes(Owner_Type, Price,color=Owner_Type))+geom_boxplot()+geom_jitter(alpha=0.1)+scale_y_log10()+xlab('Owner_Type') +ylab('Price of the data_imputed') +ggtitle('Price vs. Owner_Type')

# Price v/s Seats
data_imputed %>% ggplot(aes(as.factor(Seats), Price,color=Seats))+geom_boxplot()+geom_jitter(alpha=0.1)+scale_y_log10()+xlab('Seats') +ylab('Price of the data_imputed') +ggtitle('Price vs. seats')

# Price v/s Brand
data_imputed %>% ggplot(aes(Brand, Price,color=Brand))+geom_boxplot()+geom_jitter(alpha=0.1)+coord_flip()+scale_y_log10()+xlab('Brand') +ylab('Price of the data_imputed') +ggtitle('Price vs. Brand')

# Kilometers_Driven v/s Fuel type
data_imputed %>% ggplot(aes(Fuel_Type, Kilometers_Driven,color=Fuel_Type))+geom_boxplot()+geom_jitter(alpha=0.1)+scale_y_log10()+xlab('Fuel_Type') +ylab('Kilometers_Driven')+ggtitle('Fuel_Type vs. Kilometers_Driven')



# MODELLING

# Checking the final data dimension
dim(data_imputed)
names(data_imputed)

# Eliminating Nominal and repetitive variables before modelling
data_imputed$ID<-NULL
data_imputed$Name<-NULL
data_imputed$Location<-NULL
data_imputed$Year<-NULL
data_imputed$Brand<-NULL
data_imputed$CarModel<-NULL

dim(data_imputed)

# Scaling the numeric variables using Z score standardization
data_imputed[,c(1,5,6,8,9,11)] <- scale(data_imputed[,c(1,5,6,8,9,11)])
head(data_imputed)

# Exporting the final scaled data set
write.csv(data_imputed,"final_data_for_modelling.csv")

# Importing the final data for modelling
data_imputed<-read.csv(choose.files(),header=TRUE)
dim(data_imputed)
names(data_imputed)

# Removing the additional 'X' variable that is added while exporting by default
data_imputed$X<-NULL

# Converting all character variables to factor
data_imputed[] <- lapply(data_imputed, function(x) if(is.character(x)) as.factor(x) else x)
data_imputed$Seats<-as.factor(data_imputed$Seats)
str(data_imputed)


# Splitting the data into Train and Test set in an 80:20 ratio
set.seed(123)
sample<-sample(1:nrow(data_imputed),0.8*nrow(data_imputed))

# Train data
Train<-data_imputed[sample,]
dim(Train)

# Test data
Test<-data_imputed[-sample,] 
dim(Test)


# MODEL- 1 MULTIPLE LINEAR REGRESSION

# Building the Multiple Linear Regression model on the Train data set
model1<- lm(Price~.,data=Train)
summary(model1)

# Using Stepwise regression to arrive at the most significant variables from the data set
step_model<-step(model1)
summary(step_model)

# Predicting using the stepwise model on the Train data set
Pred_train<-predict(step_model,Train)

# Predicting using the stepwise model on the Test data set
Pred_test<-predict(step_model,Test)

# Evaluating the model performance using MAPE

# MAPE for the Train data set
mape_lm_train <- mean(abs((Pred_train - Train$Price))/Train$Price)  
mape_lm_train

# MAPE for the Test data set
mape_lm_test <- mean(abs((Pred_test - Test$Price))/Test$Price)  
mape_lm_test

# Re-building the Multiple Linear regression model after transforming the dependent variable 'Price' using log transformation
model1_lr_log<- lm(log(Price)~.,data=Train)
summary(model1_lr_log)

# Re-building the Step model with the log transformed dependent variable 'Price'
step_model_log<-step(model1_lr_log)
summary(step_model_log)

# Predicting the stepwise model on the Train data set
Pred_train_log<-exp(predict(step_model_log,Train))

# Predictin the stepwise model on the  Test data set
Pred_test_log<-exp(predict(step_model_log,Test))

# Evaluating the model performance using MAPE

# MAPE for the Train data set
mape_train_log <- mean(abs((Pred_train_log - Train$Price))/Train$Price)  
mape_train_log

# MAPE for the Test data set
mape_test_log <- mean(abs((Pred_test_log - Test$Price))/Test$Price)  
mape_test_log


# MODEL 2 - RANDOM FOREST

# Building the Random Forest model with default parameters on the Train data set
rf_model <- randomForest(formula = Price ~ .,data    = Train)
rf_model

# Tuning the model to obtain the right mtry value
tRF <- tuneRF(x = Train[,-c(10)],y=Train$Price,mtryStart = 3, ntreeTry=500,stepFactor = 1.5,improve = 0.0001, trace=TRUE,plot = TRUE,doBest = TRUE,nodesize = 100, importance=TRUE)
tRF

# Predicting the Random Forest model on the Train data set
Pred_train_rf<-predict(tRF,Train)

# Predicting the Random Forest model on the Test data set
Pred_test_rf<-predict(tRF,Test)

# Evaluating the model performance using MAPE

# MAPE for the Train data set
mape_train_rf <- mean(abs((Pred_train_rf - Train$Price))/Train$Price)  
mape_train_rf

# MAPE for the Test data set
mape_test_rf <- mean(abs((Pred_test_rf - Test$Price))/Test$Price)  
mape_test_rf


# MODEL - 3 XGBOOST

# Converting the Train and Test data set to the required matrix form for XGBoost

target <- Train$Price
predictor.matrix <- model.matrix(Price~., Train)
train.predictor.matrix <- model.matrix(Price~., Train)
test.predictor.matrix <- model.matrix(Price~., Test)

# Performing a grid search to find the optimal hyperparameter values
gs.model.xgb <-train(predictor.matrix[,-1],target,tuneGrid = expand.grid(nrounds=500,max_depth=c(3,4,5),eta=c(1e-1, 1e-2, 1e-3),gamma=0,min_child_weight=1,colsample_bytree=1,subsample=1),trControl = trainControl(method = "cv",verboseIter = TRUE),method = "xgbTree",verbose = TRUE)

# Viewing the optimal tuned hyperparameter values
model.xgb <- gs.model.xgb$finalModel
gs.model.xgb$bestTune %>% t()

# Model performance metrics
gs.model.xgb$results %>% rownames_to_column() %>% filter(rowname==rownames(gs.model.xgb$bestTune)) %>% t()

# Predicting the XGBoost model on the Train data set
train.response.xgb <- predict(model.xgb, xgb.DMatrix(train.predictor.matrix[,-1]))

# Predicting the XGBoost model on the Test data set
colnames(test.predictor.matrix) <- NULL
test.response.xgb <- predict(model.xgb, xgb.DMatrix(test.predictor.matrix[,-1]))

# Evaluating the model performance using MAPE

# MAPE for the Train data set
mape_train_xgb <- mean(abs((train.response.xgb - Train$Price))/Train$Price)  
mape_train_xgb

# MAPE for the Test data set
mape_test_xgb <- mean(abs((test.response.xgb - Test$Price))/Test$Price)  
mape_test_xgb


# PREDICTION ON THE ACTUAL TEST DATA SET (Without Price) USING THE BEST MODEL(Multiple Linear Regression)

# Loading the test data set(Without Price) 
test_noY<-read.csv(choose.files(),header = TRUE)

# test data dimension
dim(test_noY)

# Pre-processing the test data set using all the steps that were used for the data set with the dependent variable 'Price'

# Due to the presence of units, R is not able to identify all missing values. Hence we replaced  all blanks with the word 'missing' in the csv file and then imported it.
# Replacing the word 'missing' with NA
test_noY[test_noY=='missing']<-NA
View(test_noY)

# Checking for missing values
plot_intro(test_noY)
sum(is.na(test_noY))
colSums(is.na(test_noY))/nrow(test_noY)*100

# Separating units from numeric variables 
test_noY$Engine <- gsub(" CC","",test_noY$Engine)
test_noY$Engine <- as.integer(test_noY$Engine)
test_noY$Power <- gsub(" bhp","",test_noY$Power)
test_noY$Power <- as.integer(test_noY$Power)

# Eliminating the New_Price variable since it has 85% (>30%) of missing values
test_noY$New_Price<-NULL

# Since the two units under Mileage represent two different fuel types (km/kg for CNG or LPG cars) and (kmpl for Petrol or Diesel cars) we Split 'Mileage' into two variables 'kmpkg' and 'kmpl' and eliminated 'Mileage'
test_noY %>%mutate(Mileage = as.numeric(str_extract(Mileage, "^[:graph:]+"))) %>%mutate(kmpkg = ifelse(Fuel_Type=="CNG" | Fuel_Type=="LPG", Mileage, 0)) %>%mutate(kmpL = ifelse(Fuel_Type=="Diesel" | Fuel_Type=="Petrol", Mileage, 0)) %>%select(-Mileage) -> test_noY

# Re-checking for missing values
colSums(is.na(test_noY))

# Imputing the remaining missing values under 'Engine', 'Power' and 'Seats' using knn imputation

# Converting all character variables to factor for imputation
test_noY[] <- lapply(test_noY, function(x) if(is.character(x)) as.factor(x) else x)
test_noY$Seats<-as.factor(test_noY$Seats)

# Knn imputation using 3 nearest neighbours
data_imputed_test_noY = knnImputation(test_noY,k=3)

# Rechecking for missing values
sum(is.na(data_imputed_test_noY))
names(data_imputed_test_noY)

# Creating a new variable 'Age_of_car' from 'Year'
data_imputed_test_noY$Age_of_car <- as.integer(2021-test_noY$Year)

dim(data_imputed_test_noY)

# Checking for duplicate records in the data set
sum(duplicated(data_imputed_test_noY$ID))

# Eliminating Nominal and repetitive variables 
data_imputed_test_noY$X<-NULL
data_imputed_test_noY$Name<-NULL
data_imputed_test_noY$Location<-NULL
data_imputed_test_noY$Year<-NULL

dim(data_imputed_test_noY)
names(data_imputed_test_noY)
str(data_imputed_test_noY)

# Scaling the numeric/continuous variables using Z score standardization
data_imputed_test_noY[,c(1,5,6,8,9,10)] <- scale(data_imputed_test_noY[,c(1,5,6,8,9,10)])
head(data_imputed_test_noY)

# Predicting on the test data using the Multiple Linear Regression model (step model)
Pred_test_noY_log<-exp(predict(step_model_log,data_imputed_test_noY))
Pred_test_noY_log

# Converting the predictions to a data frame
Pred_test_noY_log<-as.data.frame(Pred_test_noY_log)

# Exporting the final predictions
write.csv(Pred_test_noY_log,"Final_Price_Predictions.csv")


