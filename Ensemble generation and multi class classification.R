##앙상블##
##########
library(mlr)
library(caret)
library(caretEnsemble) #앙상블
library(caTools)
library(MASS)
library(earth) #다변량 적응 회귀 스플라인
library(methods)
library(xgboost)
library(randomForest)
pima <- rbind(Pima.tr, Pima.te)
set.seed(502)
split <- createDataPartition(y = pima$type, p = 0.75, list = F) #데이터 분리
train <- pima[split, ]
test <- pima[-split, ]

table(train$type)
#희소 데이터 클래스의 데이터를 업샘플링(upsamling)
#풍부한 데이터 클래스의 데이터를 다운샘플링, 합성된 데이터
#업샘플링
control <- trainControl(method = "cv", number = 5, savePredictions = "final",
                        classProbs = T, index = createResample(train$type, 5),
                        sampling = "up", summaryFunction = twoClassSummary)

#모델학습#분류트리_다변량회귀스플라인_K-최근접이웃법
#algorithmList <- c("xgbLinear","glm", "svmRadial", "rpart", "earth", "knn", "rf", "lm",
#                   "mlpWeightDecay", "blackboost", "parRF", "nnet)
#metric <- c("ROC","RMSE")

set.seed(2)
models <- caretEnsemble::caretList(type ~., data = train, trControl = control,
                              metric = "ROC", methodList = c("rf", "xgbLinear", "svmRadial"))
models
modelCor(resamples(models)) #모델 간 상관관계

model_preds <- lapply(models, predict, newdata=test, type="prob")
model_preds <- lapply(model_preds, function(x) x[,"Yes"])
model_preds <- data.frame(model_preds)

#모형쌓기#5개의 부트스트랩된 표본에 관한 로지스틱 회귀
stack <- caretStack(models, method = "glm", metric = "ROC",
                    trControl = trainControl(method = "boot", number = 5,
                                             savePredictions = "final", classProbs = TRUE,
                                             summaryFunction = twoClassSummary))
summary(stack)

prob <- 1 - predict(stack, newdata = test, type = "prob")
model_preds$ensemble <- prob
colAUC(model_preds, test$type)


##다중 클래스 분류##
####################
library(mlr) #리샘플링
library(ggplot2)
library(HDclassif)
library(DMwR)
library(reshape2)
library(corrplot)
data(wine)
table(wine$class)

#down sampling
wine$class <- as.factor(wine$class)
set.seed(11)
df <- SMOTE(class ~., wine, perc.over = 300, perc.under = 300)
table(df$class)

#서로 다른 비율 -> 평균_0 / 표준편차_1
wine.scale <- data.frame(scale(wine[, 2:5]))
wine.scale$class <- wine$class
wine.melt <- melt(wine.scale, id.vars = "class")
ggplot(data = wine.melt, aes(x = class, y = value)) + 
  geom_boxplot() +
  facet_wrap( ~ variable, ncol = 2)

#이상치_큰 값들은 75번째 백분위수로 변경, 작은 값들을 25번째 백분위수로 바꿔주는 함수
outHigh <- function(x) {
  x[x > quantile(x, 0.99)] <- quantile(x, 0.75)
  x
}

outLow <- function(x) {
  x[x < quantile(x, 0.01)] <- quantile(x, 0.25)
  x
}

wine.trunc <- data.frame(lapply(wine[,-1], outHigh))
wine.trunc <- data.frame(lapply(wine.trunc, outLow))
wine.trunc$class <- wine$class

boxplot(wine.trunc$V3 ~ wine.trunc$class)
c <- cor(wine.trunc[, -14])
corrplot.mixed(c, upper = "ellipse")

#mlr 패키지로_데이터셋 분리
library(caret)
set.seed(502)
split <- createDataPartition(y = df$class, p = 0.7, list = F)
train <- df[split, ]
test <- df[-split, ]
wine.task <- makeClassifTask(id = "wine", data = train, target = "class")

#랜덤 포레스트 
str(getTaskData(wine.task))
rdesc <- makeResampleDesc("Subsample", iters = 3)
param <- makeParamSet(
  makeDiscreteParam("ntree", values = c(750, 1000, 1250, 1500, 1750, 2000))
)
ctrl <- makeTuneControlGrid()

#하이퍼파라미터 조절
tuning <- tuneParams("classif.randomForest", task = wine.task,
                     resampling = rdesc, par.set = param, control = ctrl)
tuning$x
tuning$y

rf <- setHyperPars(makeLearner("classif.randomForest", predict.type = "prob"),
                   par.vals = tuning$x)

#학습
fitRF <- train(rf, wine.task)
fitRF$learner.model

#테스트 세트
predRF <- predict(fitRF, newdata = test)
getConfMatrix(predRF)
performance(predRF, measures = list(mmce, acc))


##능형 회귀 분석##
##################
install.packages("penalized")
library(penalized)
ovr <- makeMulticlassWrapper("classif.penalized", mcw.method = "onevsrest")
bag.ovr = makeBaggingWrapper(ovr, bw.iters = 10, bw.replace = TRUE, 
                             bw.size = 0.7, bw.feats = 1)

set.seed(317)
fitOVR <- mlr::train(bag.ovr, wine.task)
predOVR <- predict(fitOVR, newdata = test)

getConfMatrix(predOVR)


##MLR에서의 앙상블##
####################
pima <- rbind(Pima.tr, Pima.te)
set.seed(502)
split <- createDataPartition(y = pima$type, p = 0.75, list = F) #데이터 분리
train <- pima[split, ]
test <- pima[-split, ]
pima.task <- mlr::makeClassifTask(id = "pima", data = train, target = "type")

pima.smote <- smote(pima.task, rate = 2, nn = 3)
str(getTaskData(pima.smote))

base <- c("classif.randomForest", "classif.qda", "classif.glmnet")
learns <- lapply(base, makeLearner)
learns <- lapply(learns, setPredictType, "prob")

sl <- makeStackedLearner(base.learners = learns, super.learner = "classif.logreg",
                         predict.type = "prob", method = "stack.cv")

slFit <- mlr::train(sl, pima.smote)
predFit <- predict(slFit, newdata = test)
getConfMatrix(predFit)
performance(predFit, measures = list(mmce, acc, auc))