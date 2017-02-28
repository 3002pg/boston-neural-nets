library(MASS)
library(caret)
library(neuralnet)
library(plyr)
data <- Boston

#check for missing values
apply(data, 2, function(x) sum(is.na(x)))

#split the data into train and test
set.seed(500)
index <- createDataPartition(data$medv, p=0.75, list=F)
train <- data[index,]
test <- data[-index,]

#fit a linear model as benchmark with 10-fold CV
train.control <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE
)

mod.lm <- train(medv~.,
                data=train,
                method = "lm",
                trControl = train.control)

summary(mod.lm)
pred.lm <- predict(mod.lm, test)
RMSE.lm <- sqrt(mean((pred.lm - test$medv)^2))


#data preprocessing for neural nets
#scale and split the data

maxs <- apply(data, 2, max)
mins <- apply(data, 2, min)

scaled <- as.data.frame(scale(data, center = mins, scale = maxs))
ind <- createDataPartition(scaled$medv, p = 0.75, list=F)
train_sc <- scaled[ind,]
test_sc <- scaled[-ind,]

#paste the nnet formula
n <- names(train)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))

#intialize cross-val
set.seed(450)
cv.error <- NULL
k <- 10

#create progress bar
pbar <- create_progress_bar('text')
pbar$init(k)

for(i in 1:k){
  index <- sample(1:nrow(train_sc), round(0.9*nrow(train_sc)))
  train.cv <- train_sc[index,]
  test.cv <- train_sc[-index,]
  
  mod.nn <- neuralnet(f, data=train.cv, hidden = c(5,2), linear.output = TRUE)
  
  pr.nn <- compute(mod.nn, test.cv[,1:13])
  
  #descale pred and test.cv objects
  pr.nn <- pr.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
  test.cv.r <- (test.cv$medv)*(max(data$medv)-min(data$medv))+min(data$medv)
  
  cv.error[i] <- sqrt(mean((pr.nn - test.cv.r)^2)) 
  
  pbar$step()
}

#validation error
RMSE.nn.cv <- mean(cv.error)

#compute test error
pred.nn <- compute(mod.nn, test_sc[,1:13])

pred.nn <- pred.nn$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test.sc <- (test_sc$medv)*(max(data$medv)-min(data$medv))+min(data$medv)

RMSE.nn.test <- sqrt(mean((pred.nn - test.sc)^2))
print(paste(RMSE.lm, RMSE.nn.cv, RMSE.nn.test))



