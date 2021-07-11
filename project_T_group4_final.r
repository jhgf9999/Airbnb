#load libraries
library(tidyverse)
library(caret)
library(tree)
library(class)
library(glmnet)
library(ROCR) 
library(stringr)
library(tm)
library(RTextTools)
library(xgboost)
library(randomForest)
library(ISLR)
library(gbm)

#accuracy function
accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

#load data files
train_x <- read_csv("airbnb_train_x_2021.csv")
train_y <- read_csv("airbnb_train_y_2021.csv")
test_x <- read_csv("airbnb_test_x_2021.csv")

airbnb <- read.csv("AB_US_2020.csv")
airbnb_train <- airbnb[1:100000,c(12,14)]

airbnb_test <- airbnb[100001:112199, c(12,14)]

airbnb_train <- airbnb_train %>%
  mutate(number_of_reviews=ifelse(is.na(number_of_reviews), mean(number_of_reviews, na.rm=TRUE), number_of_reviews),
         reviews_per_month=ifelse(is.na(reviews_per_month), mean(reviews_per_month, na.rm=TRUE), reviews_per_month))
airbnb_test <- airbnb_test %>%
  mutate(number_of_reviews=ifelse(is.na(number_of_reviews), mean(number_of_reviews, na.rm=TRUE), number_of_reviews),
         reviews_per_month=ifelse(is.na(reviews_per_month), mean(reviews_per_month, na.rm=TRUE), reviews_per_month))


#ALCH: bind the train.x to the test.x first
combined_data <- rbind(train_x, test_x)

#feature engineering for combined data
combined_data2 <- combined_data %>%
  select(-X1,-access,-city,-city_name,-country,-description, -experiences_offered,-host_about,-host_acceptance_rate, -host_location, -host_name,
         -host_neighbourhood,-first_review,-host_since,-host_total_listings_count, -house_rules,-interaction,-license,-monthly_price,-name,-neighborhood_overview,
         -notes,-smart_location,-space, -square_feet,-street,-summary,-weekly_price, -zipcode, -jurisdiction_names, -neighbourhood,
         -transit, -longitude, -latitude) %>%
  mutate(accommodates = ifelse(is.na(accommodates),mean(accommodates, na.rm=TRUE), accommodates),
         availability_30= ifelse(is.na(availability_30), mean(availability_30, na.rm=TRUE), availability_30),
         availability_365= ifelse(is.na(availability_365), mean(availability_365, na.rm=TRUE), availability_365),
         availability_60= ifelse(is.na(availability_60), mean(availability_60, na.rm=TRUE), availability_60),
         availability_90= ifelse(is.na(availability_90), mean(availability_90, na.rm=TRUE), availability_90),
         bathrooms=ifelse(is.na(bathrooms), mean(bathrooms, na.rm=TRUE), bathrooms),
         bed_type = ifelse(bed_type=="100%"|bed_type=="81%" | is.na(bed_type), "Other", bed_type),
         bed_type=as.factor(bed_type),
         bedrooms= ifelse(is.na(bedrooms), round(mean(bedrooms, na.rm=TRUE),1), bedrooms),
         beds = ifelse(is.na(beds), mean(beds, na.rm=TRUE), beds),
         cancellation_policy = as.factor(case_when(
           cancellation_policy %in% c("1,0", "2.0","5.0","Other") ~"Others",
           cancellation_policy %in% c("strict", "super_strict_30","super_strict_60") ~"Strict",
           cancellation_policy %in% c("no_refunds") ~"No Refunds",
           cancellation_policy %in% c("moderate") ~"Moderate",
           cancellation_policy %in% c("flexible") ~"Flexible",
           TRUE ~"Unknown")),
         cleaning_fee = as.numeric(gsub("\\$", "", cleaning_fee)),
         cleaning_fee= ifelse(is.na(cleaning_fee), round(mean(cleaning_fee, na.rm=TRUE),2), cleaning_fee),
         country_code = as.factor(ifelse(country_code!="US"|is.na(country_code), "Other",country_code)),
         extra_people = as.numeric(gsub("\\$", "", extra_people)),
         extra_people = ifelse(is.na(extra_people), round(mean(extra_people, na.rm=TRUE),2), extra_people),
         guests_included = ifelse(guests_included<0,0, guests_included),
         guests_included = ifelse(is.na(guests_included), mean(guests_included, na.rm=TRUE), guests_included),
         host_has_profile_pic = as.factor(ifelse(is.na(host_has_profile_pic), "Unknown", host_has_profile_pic)),
         host_identity_verified = as.factor(ifelse(is.na(host_identity_verified),"Unknown", host_identity_verified)),
         host_is_superhost =as.factor(ifelse(is.na(host_is_superhost), "Unknown", host_is_superhost)),
         host_listings_count = ifelse(is.na(host_listings_count), mean(host_listings_count, na.rm=TRUE), host_listings_count),
         host_listings_count=as.factor(case_when(host_listings_count<2~"1",
                                                 host_listings_count<5~"2-4",
                                                 host_listings_count<50~"5-49",
                                                 host_listings_count<200~"50-199",
                                                 TRUE~">200")),
         host_response_rate=as.numeric(gsub("%","",host_response_rate)),
         host_response_rate= ifelse(is.na(host_response_rate), round(mean(host_response_rate, na.rm=TRUE),2), 
                                    host_response_rate),
         host_response_time=case_when(host_response_time=="within an hour"~"<1",
                                      host_response_time=="within a few hours"~"<10",
                                      host_response_time=="within a day"~"<24",
                                      host_response_time=="a few days or more"~">24",
                                      TRUE~"Others"),
         host_response_time=as.factor(host_response_time),
         instant_bookable=as.factor(ifelse(is.na(instant_bookable),"Unkown",instant_bookable)),
         is_business_travel_ready=as.factor(ifelse(is.na(is_business_travel_ready),"Unkown",is_business_travel_ready)),
         is_location_exact=as.factor(ifelse(is.na(is_location_exact),"Unkown",is_location_exact)),
         maximum_nights=ifelse(is.na(maximum_nights),median(maximum_nights,na.rm=TRUE),maximum_nights),
         minimum_nights=ifelse(is.na(minimum_nights),median(minimum_nights,na.rm=TRUE),minimum_nights),
         minimum_nights=ifelse(minimum_nights>100,100,minimum_nights),
         property_type=ifelse(is.na(property_type),"Other",property_type),
         property_type=case_when(property_type %in% c("Aparthotel", "Barn", "Casa particular (Cuba)","Cave","Earth house",
                                                      "Earth House","Farm stay","Island","Lighthouse","Nature lodge","Plane",
                                                      "Tiny house", "Tipi","Train","Chalet","Cottage","Hut","Yurt") ~ "Other_type",
                                 property_type=="Bed and breakfast" ~ "Bed & Breakfast",
                                 property_type=="Boutique hotel" ~"Hotel",
                                 property_type=="Hotel" ~ "Hotel",
                                 property_type=="Hostel" ~"Hotel",
                                 property_type=="Serviced apartment" ~"Apartment",
                                 TRUE ~ property_type),
         property_type=as.factor(property_type),
         require_guest_phone_verification=as.factor(ifelse(is.na(require_guest_phone_verification), 
                                                           "Unknown", require_guest_phone_verification)),
         require_guest_profile_picture=as.factor(ifelse(is.na(require_guest_profile_picture), 
                                                        "Unknown", require_guest_profile_picture)),
         requires_license=as.factor(ifelse(is.na(requires_license), "Unknown", requires_license)),
         room_type=ifelse(room_type=="Entire home/apt","Entire home",room_type),
         room_type=as.factor(ifelse(is.na(room_type), "Others", room_type)),
         state=ifelse(is.na(state),"Others",state),
         state=case_when(state=="Baja California"~"CA",
                         state=="ca"~"CA",
                         state=="Ca"~"CA",
                         state=="il" ~"IL",
                         state=="ny"~"NY",
                         state=="Ny"~"NY",
                         state=="New York"~ "NY",
                         state=="MP" ~ "Others",
                         state=="secc Terrazas" ~"Others",
                         TRUE~state),
         state=as.factor(state),
         market=ifelse(is.na(market)|market=="$1,100.00"|market=="$2,999.00"|market=="$750.00","Other market",market),
         market=case_when(market %in% c("Adirondacks","Agra","Atlanta","Bristol","Catskills and Hudson Valley",
                                        "Chico","Coastal Orange County","College Station","Cuba","Dallas",
                                        "Flims","Fresno","Houston","Indianapolis","Jamaica South Coast",
                                        "Lagos, NG","Las Vegas","London","Miami","Nice","Oregon Coast",
                                        "Palm Springs Desert","Paris","Philadelphia","Pittsburg","Providence",
                                        "San Antonio, US","South Florida Gulf Coast","Temecula Valley",
                                        "Toronto","Tuscany Countryside","Umbria Countryside","Venice") ~"Other market",
                          market=="Portland, Maine"~"Portland",
                          TRUE ~ market),
         market=as.factor(market),
         price=as.numeric(gsub("\\$","",price)),
         price= ifelse(is.na(price), round(mean(price, na.rm=TRUE),2), price),
         security_deposit=as.numeric(gsub("\\$","",security_deposit)),
         security_deposit= ifelse(is.na(security_deposit), round(mean(security_deposit, na.rm=TRUE),2), security_deposit),
         amenities = gsub("[{}]","", amenities),
         amenities = gsub("\"", "", amenities),
         amenities_list = strsplit(amenities, ",", fixed=TRUE),
         host_verifications = gsub("[[]]", "", host_verifications),
         host_verifications =gsub("[['']","", host_verifications),
         host_verifications =gsub("[]]","", host_verifications),
         host_verifications_list=strsplit(host_verifications,",", fixed = TRUE),
         amenities_TV = as.factor(ifelse(grepl("TV", amenities_list) | 
                                           grepl("Calbe", amenities_list), TRUE, FALSE)),
         amennities_wifi = as.factor(ifelse(grepl("Wifi", amenities_list) | 
                                              grepl("Internet", amenities_list), TRUE, FALSE)),
         amennities_air_conditioning = as.factor(ifelse(grepl("Air conditioning", amenities_list), TRUE, FALSE)),
         amennities_heating = as.factor(ifelse(grepl("Heating", amenities_list), TRUE, FALSE)),
         amennities_parking = as.factor(ifelse(grepl("Parking", amenities_list) | 
                                                 grepl("parking", amenities_list), TRUE, FALSE)),
         amennities_pet = as.factor(ifelse(grepl("Pet", amenities_list) | 
                                             grepl("pet", amenities_list), TRUE, FALSE)),
         amennities_kitchen = as.factor(ifelse(grepl("Kitchen", amenities_list) | 
                                                 grepl("kitchen", amenities_list), TRUE, FALSE)),
         host_phone = as.factor(ifelse(grepl("phone", host_verifications_list), TRUE, FALSE)),
         host_email = as.factor(ifelse(grepl("email", host_verifications_list), TRUE, FALSE)),
         host_review = as.factor(ifelse(grepl("reviews", host_verifications_list), TRUE, FALSE)),
         host_government = as.factor(ifelse(grepl("government", host_verifications_list), TRUE, FALSE)))

combined_data3 <- combined_data2 %>%
  select(-amenities, -amenities_list, -host_verifications, -host_verifications_list)

train_dat <- combined_data3[1:100000,]
test_dat <- combined_data3[100001:112199,]

tf_house_rules <- create_matrix(train_x$house_rules,
                                language="english", 
                                removeStopwords=TRUE, 
                                removeNumbers=TRUE, 
                                stemWords=TRUE, 
                                removeSparseTerms = 0.9, 
                                stripWhitespace=TRUE, 
                                toLower=TRUE)
house_rules_DF <- data.frame(as.matrix(tf_house_rules), stringsAsFactors = FALSE)
tf_name <- create_matrix(train_x$name, 
                         language="english", 
                         removeStopwords=TRUE, 
                         removeNumbers=TRUE, 
                         stemWords=TRUE, 
                         removeSparseTerms = 0.97, 
                         stripWhitespace=TRUE, 
                         toLower=TRUE)
name_DF <- data.frame(as.matrix(tf_name), stringsAsFactors = FALSE)
tf_transit <- create_matrix(train_x$transit, 
                            language="english", 
                            removeStopwords=TRUE, 
                            removeNumbers=TRUE, 
                            stemWords=TRUE, 
                            removeSparseTerms = 0.9, 
                            stripWhitespace=TRUE, 
                            toLower=TRUE)
transit_DF <- data.frame(as.matrix(tf_transit), stringsAsFactors = FALSE)
tf_summary <- create_matrix(train_x$summary, 
                            language="english", 
                            removeStopwords=TRUE, 
                            removeNumbers=TRUE, 
                            stemWords=TRUE, 
                            removeSparseTerms = 0.85, 
                            stripWhitespace=TRUE, 
                            toLower=TRUE)
summary_DF <- data.frame(as.matrix(tf_summary), stringsAsFactors = FALSE)

train_combine <- cbind(train_dat, house_rules_DF, name_DF,transit_DF, summary_DF, airbnb_train)
train_combine <- train_combine[, !duplicated(colnames(train_combine))]
rownames(train_combine) <- NULL

tf_house_rules1 <- create_matrix(test_x$house_rules,
                                 language="english", 
                                 removeStopwords=TRUE, 
                                 removeNumbers=TRUE, 
                                 stemWords=TRUE, 
                                 removeSparseTerms = 0.9, 
                                 stripWhitespace=TRUE, 
                                 toLower=TRUE)
house_rules_DF1 <- data.frame(as.matrix(tf_house_rules1), stringsAsFactors = FALSE)
tf_name1 <- create_matrix(test_x$name, 
                          language="english", 
                          removeStopwords=TRUE, 
                          removeNumbers=TRUE, 
                          stemWords=TRUE, 
                          removeSparseTerms = 0.97, 
                          stripWhitespace=TRUE, 
                          toLower=TRUE)
name_DF1 <- data.frame(as.matrix(tf_name1), stringsAsFactors = FALSE)
tf_transit1 <- create_matrix(test_x$transit, 
                             language="english", 
                             removeStopwords=TRUE, 
                             removeNumbers=TRUE, 
                             stemWords=TRUE, 
                             removeSparseTerms = 0.9, 
                             stripWhitespace=TRUE, 
                             toLower=TRUE)
transit_DF1 <- data.frame(as.matrix(tf_transit1), stringsAsFactors = FALSE)
tf_summary1 <- create_matrix(test_x$summary, 
                             language="english", 
                             removeStopwords=TRUE, 
                             removeNumbers=TRUE, 
                             stemWords=TRUE, 
                             removeSparseTerms = 0.85, 
                             stripWhitespace=TRUE, 
                             toLower=TRUE)
summary_DF1 <- data.frame(as.matrix(tf_summary1), stringsAsFactors = FALSE)

test_combine <- cbind(test_dat,house_rules_DF1, name_DF1, transit_DF1, summary_DF1, airbnb_test)
test_combine <- test_combine[, !duplicated(colnames(test_combine))]
rownames(test_combine) <- NULL

traindummy <- dummyVars( ~ .+price:security_deposit+price:cleaning_fee+price:accommodates+price:extra_people, data = train_combine)
traincombined_dummy <- data.frame(predict(traindummy, newdata = train_combine))
train_combined_dummy=traincombined_dummy[,!grepl("*FALSE",names(traincombined_dummy))]


testdummy <- dummyVars( ~ .+price:security_deposit+price:cleaning_fee+price:accommodates+price:extra_people, data = test_combine)
testcombined_dummy <- data.frame(predict(testdummy, newdata = test_combine))
test_combined_dummy=testcombined_dummy[,!grepl("*FALSE",names(testcombined_dummy))]


#ALCH: combine original y for after-dummy train data, and set y as a factor
train_dummy_y <- cbind(train_combined_dummy, train_y)
train_dummy_y <- train_dummy_y %>%
  mutate(perfect_score = as.factor(perfect_score))

# create a small dataset for training data
small <- sample(nrow(train_dummy_y), 30000)
train_dummy_y <- train_dummy_y[small,]

#then calculate the training/validation row numbers and split
va_inst <- sample(nrow(train_dummy_y), .3*nrow(train_dummy_y))
train<- train_dummy_y[-va_inst,]
valid<- train_dummy_y[va_inst,]

#tree
mycontrol = tree.control(nrow(train), mincut = 5, minsize = 10, mindev = 0.0005)
full_tree = tree(perfect_score ~ .,control = mycontrol, train)
summary(full_tree)

#define a vector of pruned tree sizes
sizevec <- c(2,4,6,8,10,15,20,25,30,35,40,45)
numsizes <- length(sizevec)

#define vectors to store the training and validation accuracy
tr1_acc <- rep(0, numsizes)
va1_acc <- rep(0, numsizes)

#loop over each size in sizevec
for (i in c(1:numsizes)){
  
  #retrieve the size indexed by i
  treesize <- sizevec[i]
  
  #prune the tree to be treesize
  pruned_tree=prune.tree(full_tree, best = treesize)
  
  #get the training accuracy
  tree_tr <- predict(pruned_tree, newdata = train)[,2]
  tree_class_tr <- ifelse(tree_tr > .5, 1, 0)
  acc_tr1 <- accuracy(tree_class_tr, train$perfect_score)  
  tr1_acc[i] <- acc_tr1 #store in tr_acc vector
  #get the validation accuracy
  tree_va <- predict(pruned_tree, newdata = valid)[,2]
  tree_class_va <- ifelse(tree_va > .5, 1, 0)
  acc_va1 <- accuracy(tree_class_va, valid$perfect_score)
  va1_acc[i] <- acc_va1 #store in va_acc vector
}

#plot line graphs
plot(sizevec, tr1_acc, col = "blue", type = 'l')
lines(sizevec, va1_acc, col = "red")

#store the predictions from the best pruning value (33)
pruned.best = prune.tree(pruned_tree, best = 33)
best.tree.preds <- predict(pruned.best, newdata = valid)[,2]

crossvalidated_trees <- cv.tree(full_tree,  , prune.misclass, K = 10)
crossvalidated_trees

#knn
train.X <- train %>%
  select(-perfect_score)

valid.X <- valid %>%
  select(-perfect_score)

train.y <- train$perfect_score
valid.y <- valid$perfect_score

train.y1=as.numeric(train.y) - 1
valid.y1=as.numeric(valid.y) - 1

best1 <- c(2,4,6,8,10,15,20)
knn_acc <- rep(0,length(best1))
for (i in best1){
  knn.pred=knn(train.X,valid.X,train.y1,k=i)
  acc <- accuracy(knn.pred, valid.y1)
  knn_acc[i] <- acc
}
knn_acc <- knn_acc[!is.na(knn_acc)&!knn_acc==0]
plot(best1, knn_acc, col="red")

best.knn.preds = knn(train.X, valid.X, train.y1, k=20)
best.knn.preds = as.numeric(best.knn.preds)-1
knn_class <- ifelse(best.knn.preds>0.5,1,0)
knn_accuacy = accuracy(best.knn.preds, valid.y1)
knn_accuacy

#xboost

xgb = xgboost(data = as.matrix(train[-220]), #Only select the features, and it must be matrix
              label = train$perfect_score, #Dependent variable (y)
              nrounds = 150,) #implement iterations

xgb_pred = predict(xgb, newdata = as.matrix(valid[-220]))
xgb_pred
xgb_pred = ifelse(xgb_pred > 1.292, 1, 0)
table(xgb_pred, valid$per)
xgb_acc <- mean(ifelse(xgb_pred == valid$perfect_score, 1, 0))
xgb_acc

#alternative - xgboost
d_train = xgb.DMatrix(data = as.matrix(train[-220]),
                     label = train$perfect_score)
d_valid = xgb.DMatrix(data = as.matrix(valid[-220]),
                    label = valid$perfect_score)

xgb.params = list(
  colsample_bytree = 1,
  subsample = 1,                      
  booster = "gbtree",
  max_depth = 12,
  eta = 0.03,
  eval_metric = "rmse",                      
  objective = "reg:linear",
  gamma = 0)

xgb.cv.model = xgb.cv(params = xgb.params, 
                      data = d_train,
                      nfold = 5, # 5-fold cv
                      nrounds = 200,
                      early_stopping_rounds = 30,
                      print_every_n = 20
) 

best.nrounds = xgb.cv.model$best_iteration 
best.nrounds

tmp = xgb.cv.model$evaluation_log

plot(x=1:nrow(tmp), y= tmp$train_rmse_mean, col='red', xlab="nround", ylab="rmse", main="Avg.Performance in CV") 
points(x=1:nrow(tmp), y= tmp$test_rmse_mean, col='blue') 
legend("topright", pch=1, col = c("red", "blue"), 
       legend = c("Train", "Validation") )

xgb.model = xgb.train(paras = xgb.params, 
                      data = d_train,
                      nrounds = best.nrounds) 
xgb_pred = predict(xgb, newdata = as.matrix(valid[-220]))
xgb_pred
summary(xgb_pred)
xgb_class = ifelse(xgb_pred > 1.55, 1, 0)
table(xgb_pred, valid$per)
xgb_acc <- accuracy(xgb_class, valid$perfect_score)
xgb_acc

#random forest
# tuning rf
ntree_grid <- c(500,1000,1500)
mtry_grid <- c(3, 4, 5)

for (n in ntree_grid){
  for (m in mtry_grid){
    rf_classifier <- randomForest(perfect_score~.,
                                  data=train,
                                  mtry=m, ntree=n)
    testforest <- predict(rf_classifier, newdata=valid)
    rf_acc <- mean(ifelse(testforest==valid$perfect_score,1,0))
    print(paste("Num trees = ",n,", mtry = ",m,", acc = ",rf_acc))
  }
}

rf_classifier = randomForest(perfect_score ~ .,mtry=5, ntree=1500, data=train)
testforest = predict(rf_classifier, newdata=valid)
table(testforest, valid$perfect_score) 
rf_acc <- mean(ifelse(testforest==valid$perfect_score,1,0))
rf_acc
summary(valid$perfect_score)


#need to set up numerical matrices for lassso and ridge
train.X <- train %>%
  select(-perfect_score)
valid.X <- valid %>%
  select(-perfect_score)
train.y <- train$perfect_score
valid.y <- valid$perfect_score

grid <- 10^seq(10,-4,length=100)
#setup for glmnet
k <- 5
train.X <- as.matrix(train.X)
valid.X <- as.matrix(valid.X)

#run cv.glmnet with nfolds = 5 to get the best lambda
lasso.out <- cv.glmnet(train.X, train.y, family="binomial", alpha=1,lambda=grid, nfolds=k)
lassolam <- lasso.out$lambda.min
lassolam
coeffs <- coef(lasso.out, s="lambda.min")
coeffs
## [1] 0.001353048
#make predictions using this lambda in the validation data
best.lasso.preds <- predict(lasso.out, s=lassolam, newx = valid.X, type="response")
lambdas <- lasso.out$lambda
errors <- lasso.out$cvm
plot(log(lambdas), log(errors))
#classify and compute accuracy
lasso_class <- ifelse(best.lasso.preds > .55, 1, 0)
accuracy(lasso_class, valid.y)


#ridge
ridge.out <- cv.glmnet(train.X, train.y, family="binomial", alpha=0, lambda=grid, nfolds=k)
ridgelam <- ridge.out$lambda.min
ridgelam
## [1] 0.01830738
coeffs1 <- coef(ridge.out, s="lambda.min")
coeffs1

#make predictions in validation data
best.ridge.preds <- predict(ridge.out, s=ridgelam, newx = valid.X, type="response")
ridlambdas <- ridge.out$lambda
riderrors <- ridge.out$cvm
plot(log(ridlambdas), log(riderrors))
#classify and compute accuracy
ridge_class <- ifelse(best.ridge.preds > 0.55, 1, 0)
accuracy(ridge_class, valid.y)


# generate ROCR performance object retrieving TPR and FPR
#tree
pred_tree <- prediction(best.tree.preds, valid.y)
perf_tree <- performance(pred_tree, "tpr", "fpr")
plot(perf_tree, col = "red")
performance(pred_tree, measure = "auc")@y.values[[1]]
#lasso
pred_lasso <- prediction(best.lasso.preds, valid.y)
perf_lasso <- performance(pred_lasso, "tpr", "fpr")
plot(perf_lasso, add = TRUE, col = "green")
performance(pred_lasso, measure = "auc")@y.values[[1]]
#ridge
pred_ridge <- prediction(best.ridge.preds, valid.y)
perf_ridge <- performance(pred_ridge, "tpr", "fpr")
plot(perf_ridge, add = TRUE, col = "purple")
performance(pred_ridge, measure = "auc")@y.values[[1]]
#xgboost
pred_xgboost <- prediction(xgb_pred, valid.y)
perf_xgboost <- performance(pred_xgboost, "tpr", "fpr")
plot(perf_xgboost, add =TRUE, col = "black")
performance(pred_xgboost, measure = "auc")@y.values[[1]]
#random forest
test.forest = predict(rf_classifier, type = "prob", newdata = valid)
forestpred = prediction(test.forest[,2], valid$perfect_score)
forestperf = performance(forestpred, "tpr", "fpr")
plot(forestperf, col="gold", add=TRUE)
performance(forestpred, measure = "auc")@y.values[[1]]
#knn
pred_knn <- prediction(knn_class, valid.y1)
roc_knn <- performance(pred_knn, "tpr", "fpr")
plot(roc_knn, add=T, col="yellow")
performance(pred_knn, measure = "auc")@y.values[[1]]

#set up cutoff variable 
cutoffs_rf <- data.frame(cut=forestperf@alpha.values[[1]], fpr=forestperf@x.values[[1]], 
                         tpr=forestperf@y.values[[1]])
#find the best cutoff
options(max.print=30000) #Expand the rows that R can print
cutoffs_findrf <- cutoffs_rf$fpr < 0.1 #finding the fpr that below 0.1
cutoffs_findrf
show(cutoffs_rf[231,]) #***ADJSUT INDEX HERE!***
#evaluate
test_dummy<-as.matrix(test_combined_dummy)
probs <- predict(rf_classifier, newdata=test_dummy,type='prob')[,2]
classifications <- ifelse(probs > 0.4, 1, 0) #***ADJUST CUTOFF HERE!***
classifications <- ifelse(is.na(classifications), 0, classifications)
summary(classifications)

#Prediction Output:
write.table(classifications, "predictions_group4_0506.csv", row.names = FALSE, col.names = FALSE)
