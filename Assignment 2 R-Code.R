setwd("C:\\Users\\ninas\\R\\RPackages")

.libPaths('C:\\Users\\ninas\\R\\RPackages')

#loading libraries
library(readxl)
library('psych')
library('plyr')
library('glmnet')
library(car)
library('aod')
library(MASS)
library('corrplot')
library(class)
library('mclust')
library('class')
library('e1071')
library('tree')
library(randomForest)
library('DT')
library('MLmetrics')


#read xls file
sale_attempts <- read_xls("C:\\Users\\ninas\\OneDrive\\Desktop\\MSc Business Analytics\\2nd Quarter\\Statistics for BA 2\\Assignment 1\\project I  2021-2022.xls")
sale_attempts <- data.frame(sale_attempts)
#check for nulls, NAs
sum(is.na(sale_attempts))
sum(is.null(sale_attempts))

#checking the shape of the data
str(sale_attempts)
describe(sale_attempts)
summary(sale_attempts)

#update job, marital status, education, loan default index, housing loan  index, personal loan index 
#contact index, month and weekday of last contact and campaign outcome index to factors
sale_attempts$job <- factor(sale_attempts$job, labels = 1:12)
sale_attempts$marital <- factor(sale_attempts$marital, labels = 1:4)
sale_attempts$education <- factor(sale_attempts$education, labels = 1:8)
sale_attempts$default <- factor(sale_attempts$default, labels = 1:3)
sale_attempts$housing <- factor(sale_attempts$housing, labels = 1:3)
sale_attempts$loan <- factor(sale_attempts$loan, labels = 1:3)
sale_attempts$contact <- factor(sale_attempts$contact, labels = 1:2)
sale_attempts$day_of_week <- factor(sale_attempts$day_of_week, labels = 1:5)
sale_attempts$poutcome <- factor(sale_attempts$poutcome, labels = 1:3)
#update subscription index from no/yes to 0/1 since it will be the dependent variable in the model
sale_attempts[which(sale_attempts$SUBSCRIBED == 'no'),21] <- '0'
sale_attempts[which(sale_attempts$SUBSCRIBED == 'yes'),21] <- '1'
sale_attempts$SUBSCRIBED <- as.numeric(sale_attempts$SUBSCRIBED)
unique(sale_attempts$default)


#all values of 'consumer confidence index' are negative, they are updated to positive number
sale_attempts$cons.conf.idx <-  (-1)*sale_attempts$cons.conf.idx

n <- nrow(sale_attempts)
#variable pdays has '999' in almost all of its observations, it does not make any sense to keep them like that,
#but since there is already a variable that indexes whether a customer was contacted in previous campaigns (poutcome)
#the variable pdays is dropped completely
sale_attempts <- sale_attempts[,-13]

sale_attempts$month <- factor(sale_attempts$month, labels = 1:10)
sale_attempts$SUBSCRIBED <- factor(sale_attempts$SUBSCRIBED) 

#Part 1
#We will implement 3 different classification methods

library('splitTools')
partioned <- partition(1:n, p = c(train = 0.9, val = 0.1))
valid <- partioned$val
test_and_train <- partition(partioned$train, p = c(train = 0.7, test = 0.3))
test_df <- test_and_train$test
train_df <- test_and_train$train
validation <- sale_attempts[valid, -20]
val_target <- sale_attempts[valid, 20]
# train_df <- 
#random observations from the dataset
deiktes<-sample(partioned$train)
clf_sales <- sale_attempts[deiktes,]
n <- length(deiktes)

#5 folds for cross validation
k<-5
clf_methods <- c('tree','svm', 'naiveBayes', 'forest', 'logit')
metrics <- c('accuracy', 'recall', 'precision', 'f1')
#random seed to keep consistent results
set.seed(3)
#goodness of fit measures that will be calculated for each method for in-sample predictions
in_sample <- matrix(data=NA, ncol = length(metrics), nrow = length(clf_methods))
rownames(in_sample) <- clf_methods
colnames(in_sample) <- metrics

accuracy <- matrix(data=NA, ncol= k, nrow = length(clf_methods))
rec <- matrix(data=NA, ncol= k, nrow = length(clf_methods))
precision <- matrix(data=NA, ncol= k, nrow = length(clf_methods))
f1 <- matrix(data=NA, ncol= k, nrow = length(clf_methods))
rownames(accuracy) <- rownames(precision) <- rownames(rec) <- rownames(f1) <-clf_methods

#creation of a scaled df
only_numeric <- sapply(sale_attempts, class) == 'numeric'
scaled_num <- scale(sale_attempts[,only_numeric])
scaled_df <- sale_attempts
scaled_df[,only_numeric] <- scaled_num

#convert factor variables to dummy variables
# Load the library
library(fastDummies)
scaled_df <- dummy_cols(scaled_df, 
           select_columns = c("job",'marital','education','default','housing','loan','contact','month','day_of_week','poutcome'))
scaled_df <- scaled_df[,c(-2,-3,-4,-5,-6,-7,-8,-9,-10,-14)]
scaled_df <- scaled_df[deiktes,]
cols <- c(11:63)
scaled_df[,cols] <- lapply(scaled_df[,cols], factor)
#we will fit a logistic regression on all data to find the best threshold to classify customers as buyers or non-buyers
GLModel <- glm(SUBSCRIBED ~ ., data=sale_attempts, family = 'binomial')

res<-NULL
for (threshold in seq(0.01,0.99,by=0.01)){
  clas<- GLModel$fitted > threshold
  conftable<- table(clas,sale_attempts$SUBSCRIBED)
  sens<- conftable[1,1]/apply(conftable,2,sum)[1]
  spec<- conftable[2,2]/apply(conftable,2,sum)[2]
  res<-rbind(res, c(sens,1-spec, threshold))
}

res<-res[order(res[,1]),]

plot(res[,2],res[,1], xlab="FN", ylab="TP", type="l",col=2, xlim=c(0,1), ylim=c(0,1))
abline(0,1)

distance <- ((1 - res[,1])^2 + (0 - res[,2])^2)^(1/2)
res <- cbind(res,distance)
best_threshold <- res[which.min(distance),3]; best_threshold


for (i in 1:k){
  te <- deiktes[((i-1)*(n/k)+1):(i*(n/k))]	
  train <- clf_sales[-te, ]
  test <- na.omit(clf_sales[te, -20])
  test_target <- na.omit(clf_sales[te, 20])
  
  train_sc <- scaled_df[-te,]
  test_sc <- na.omit(scaled_df[te, -10])
  test_target_sc <- na.omit(scaled_df[te, 10])

  #1st method - Decision Trees Classifier
  tree_fit <- tree(SUBSCRIBED ~ ., data = train)
  tree_pr <- predict(tree_fit,newdata=test,type='class')
  
  #in-sample evaluation
  TP <- table(tree_pr,test_target)[2,2]
  FP <- table(tree_pr,test_target)[2,1]
  FN <- table(tree_pr,test_target)[1,2]
  
  accuracy['tree',i] <- round(sum(test_target == tree_pr)/dim(test)[1],2)
  precision['tree',i] <- round(TP / (TP + FP),2)
  rec['tree',i] <- round(TP / (TP + FN),2)
  f1['tree',i] <- round(F1_Score(tree_pr,test_target, positive = 1),2)


  #2nd method - Support Vector Machines
  svm_fit <- svm(SUBSCRIBED~., data=train_sc)
  svm_pr <- predict(svm_fit, newdata=test_sc)
  
  #in-sample evaluation
  TP <- table(svm_pr,test_target_sc)[2,2]
  FP <- table(svm_pr,test_target_sc)[2,1]
  FN <- table(svm_pr,test_target_sc)[1,2]
  
  accuracy['svm',i] <- round(sum(test_target_sc == svm_pr)/dim(test)[1],2)
  precision['svm',i] <- round(TP / (TP + FP),2)
  rec['svm',i] <- round(TP / (TP + FN),2)
  f1['svm',i] <- round(F1_Score(svm_pr,test_target_sc, positive = 1),2)
  
  
  #3rd method - Naive-Bayes
  NB_fit <- naiveBayes(SUBSCRIBED ~ ., data = train)
  NB_pr <- predict(NB_fit, test)
  
  #in-sample evaluation
  TP <- table(NB_pr,test_target)[2,2]
  FP <- table(NB_pr,test_target)[2,1]
  FN <- table(NB_pr,test_target)[1,2]
  
  accuracy['naiveBayes',i] <- round(sum(test_target == NB_pr)/dim(test)[1],2)
  precision['naiveBayes',i] <- round(TP / (TP + FP),2)
  rec['naiveBayes',i] <- round(TP / (TP + FN),2)
  f1['naiveBayes',i] <- round(F1_Score(NB_pr,test_target, positive = 1),2)
 
   
  #4th method - Random Forest
  forest <- randomForest(SUBSCRIBED ~ ., data=train, ntree=500,
                         mtry=12, importance=TRUE)
  rand.for<- predict(forest, test) 
  
  
  #in-sample evaluation
  TP <- table(rand.for,test_target)[2,2]
  FP <- table(rand.for,test_target)[2,1]
  FN <- table(rand.for,test_target)[1,2]
  
  accuracy['forest',i] <- round(sum(test_target == rand.for)/dim(test)[1],2)
  precision['forest',i] <- round(TP / (TP + FP),2)
  rec['forest',i] <- round(TP / (TP + FN),2)
  f1['forest',i] <-round( F1_Score(rand.for,test_target, positive = 1),2)
  
  
  #5th method - Logistic Regression
  GLModel <- glm(SUBSCRIBED ~ ., data=train, family = 'binomial')
  fitGLM <- as.numeric(predict(GLModel,newdata=test, type = 'response') > best_threshold)
  
  #in-sample evaluation
  TP <- table(fitGLM,test_target)[2,2]
  FP <- table(fitGLM,test_target)[2,1]
  FN <- table(fitGLM,test_target)[1,2]
  
  accuracy['logit',i] <- round(sum(test_target == fitGLM)/dim(test)[1],2)
  precision['logit',i] <- round(TP / (TP + FP),2)
  rec['logit',i] <- round(TP / (TP + FN),2)
  f1['logit',i] <-round( F1_Score(fitGLM,test_target, positive = 1),2)
}

#plotting the tree
plot(tree_fit)
text(tree_fit)


#overall accuracy performance of all methods - in-sample pred
accuracy <- cbind(accuracy, apply(accuracy, 1, mean))
dimnames(accuracy)[[2]] <- c('1','2','3','4','5','mean')
in_sample[,'accuracy'] <- accuracy[,'mean']

precision <- cbind(precision, apply(precision, 1, mean))
dimnames(precision)[[2]] <- c('1','2','3','4','5','mean')
in_sample[,'precision'] <- precision[,'mean']

rec <- cbind(rec, apply(rec, 1, mean))
dimnames(rec)[[2]] <- c('1','2','3','4','5','mean')
in_sample[,'recall'] <- rec[,'mean']

f1 <- cbind(f1, apply(f1, 1, mean))
dimnames(f1)[[2]] <- c('1','2','3','4','5','mean')
in_sample[,'f1'] <- f1[,'mean']
datatable(round(in_sample,2), class = 'cell-border stripe', caption = 'Average values of each measure for in-sample predictions')


#goodness of fit measures that will be calculated for each method for out-of sample predictions
out_of_sample <- matrix(data=NA, ncol = length(metrics), nrow = 1)
rownames(out_of_sample) <- 'Random_Forest'
colnames(out_of_sample) <- metrics

#overall accuracy performance for the random forest method - out of sample pred
forest <- randomForest(SUBSCRIBED ~ ., data=clf_sales, ntree=500,
                       mtry=12, importance=TRUE)
rand.for_eval<- predict(forest, validation) 

out_of_sample[,'accuracy'] <- round(sum(val_target == rand.for_eval)/dim(validation)[1],2)
out_of_sample[,'precision'] <- round(TP / (TP + FP),2)
out_of_sample[,'recall'] <- round(TP / (TP + FN),2)
out_of_sample[,'f1'] <- round(F1_Score(rand.for_eval,val_target, positive = 1),2)
datatable(out_of_sample, class = 'cell-border stripe', caption = 'Average values of each measure for out of sample predictions')




#Part 2 - Clustering
#We will implement clustering methods to identify clusters of customers with common features.
clust_sales <- sale_attempts[,c(1,2,3,4,5,6,7,12,13,14,20)]
head(clust_sales)

n <- nrow(clust_sales)
clust_ind <- sample(1:n, 10000)
clust_sample <- clust_sales[clust_ind,]

#scaling of the clustering df
only_numeric2 <- sapply(clust_sample, class) == 'numeric'
scaled_num2 <- scale(clust_sample[,only_numeric2])
scaled_clust <- clust_sample
scaled_clust[,only_numeric2] <- scaled_num2

#calculating the distance matrix
library('cluster')
distance_matrix <- daisy(scaled_clust[,-11], metric = 'gower')

#fitting the hierarchical clustering model using the ward method
hc1 <- hclust(distance_matrix,method="ward.D")
#plotting a dendrogram of the clustering and bordering 2 clusters
plot(hc1)
rect.hclust(hc1, k = 2, border = 'red')
#examining possible classifications from 2 to 5 clusters
clust <- cutree(hc1, k = 2:5)

#based on the silhouette plot we select 4 clusters
plot(silhouette(cutree(hc1, k=2), distance_matrix),col=1:2,main='Ward', border=NA)
#ari is negative, worse than random guesses. The clusters are not associated
#with the subscription indexes of the customers
adjustedRandIndex(clust_sample$SUBSCRIBED,clust[,1])


#assign the cluster number for each customer in a new column
clust_sample$cluster <- clust[,1]
#plot interesting differences between the two clusters
par(mfrow = c(1,2))
#-----------------------------------------------------------------------------#
#Plotting the age distributions
hist(clust_sample[which(clust_sample$cluster ==1),1], col = 'lightblue'
     , main = 'Age Distribution - Cluster 1', xlab = 'Age')
hist(clust_sample[which(clust_sample$cluster ==2),1], col = 'lightblue'
     , main = 'Age Distribution - Cluster 2', xlab = 'Age')

#-----------------------------------------------------------------------------#
#Plotting the job occupation
library(ggplot2)
#creation of a dataframe containing the joint frequency of each job and cluster
temp_df <- as.data.frame(as.matrix(table(clust_sample$job, clust_sample$cluster)))
#assigning the proper labels to each job
job <- c("admin.", "blue-collar","entrepreneur","housemaid","management","retired"
         ,"self-employed","services","student","technician","unemployed", "unknown")  
#update the dataframe
temp_df[,1] <- factor(job)
colnames(temp_df) <- c('Job', 'Cluster', 'Frequency')


#count of the number of customers per cluster
clust_1_rows <- nrow(clust_sample[which(clust_sample$cluster==1),])
clust_2_rows <- nrow(clust_sample[which(clust_sample$cluster==2),])

#update the dataframe with their corresponding percentages
temp_df[which(temp_df$Cluster==1),3] <- temp_df[which(temp_df$Cluster==1),3]/clust_1_rows
temp_df[which(temp_df$Cluster==2),3] <- temp_df[which(temp_df$Cluster==2),3]/clust_2_rows

#plot the joint frequency of the jobs and the clusters
ggplot(temp_df, aes(Job, Frequency, fill = Cluster)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")

#-----------------------------------------------------------------------------#
#Plotting the education levels
#creation of a dataframe containing the joint frequency of each education level and cluster
temp_df <- as.data.frame(as.matrix(table(clust_sample$education, clust_sample$cluster)))
#assigning the proper labels to each education level
education <- c("basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate"
          , "professional.course", "university.degree", "unknown")  
#update the dataframe
temp_df[,1] <- factor(education)
colnames(temp_df) <- c('education', 'Cluster', 'Frequency')


#update the dataframe with their corresponding percentages
temp_df[which(temp_df$Cluster==1),3] <- temp_df[which(temp_df$Cluster==1),3]/clust_1_rows
temp_df[which(temp_df$Cluster==2),3] <- temp_df[which(temp_df$Cluster==2),3]/clust_2_rows

#plot the joint frequency of the education levels and the clusters
ggplot(temp_df, aes(education, Frequency, fill = Cluster)) + 
  geom_bar(stat="identity", position = "dodge") + 
  scale_fill_brewer(palette = "Set1")




      