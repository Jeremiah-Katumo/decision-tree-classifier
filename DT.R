library(tidyverse)
library(rsample)
library(caret)
library(recipes)
library(vip)
library(pdp)
library(rpart)
library(rpart.plot)

data("iris")


set.seed(42) # for reproducibility
split <- initial_split(iris, prop = 0.8, strata = "Species")
iris_train <- training(split)
iris_test <- testing(split)

blueprint <- recipe(Species ~., data = iris_train) %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

iris_train_prep = blueprint %>%
  prep(iris_train) %>%
  bake(iris_train)

iris_test_prep <- blueprint %>%
  prep(iris_test) %>%
  bake(iris_test)


iris_dt <- rpart(Species ~., 
                 data = iris_train_prep,
                 method = "anova")
iris_dt
rpart.plot(iris_dt)
plotcp(iris_dt, col = "red") # visual representation of cross-validation results in an rpart object
printcp(iris_dt)

iris_dt2 <- rpart(formula = Species ~.,
                  data = iris_train_prep,
                  method = "anova",
                  control = list(cp = 0, xval = 10))
iris_dt2
pred <- predict(iris_dt2, newdata = iris_test)

rpart.plot(iris_dt2)
plotcp(iris_dt2)


# create ames training data
set.seed(123)
ames <- AmesHousing::make_ames()
split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)

ames_dt <- rpart(formula = Sale_Price ~.,
                 data = ames_train,
                 method = "anova",
                 control = list(cp=0, minbucket = 5, maxdepth = 3)) # xval=10, 
ames_dt
rpart.plot(ames_dt)
plotcp(ames_dt)

blueprint2 <- recipe(Sale_Price ~., data = ames_train) %>%
  step_zv(all_numeric()) %>%
  # step_zv(all_nominal()) %>%
  step_impute_median(all_numeric()) %>%
  step_impute_knn(all_numeric(), neighbors = 7) %>%
  step_YeoJohnson(all_numeric()) %>%
  # step_integer(matches="Qual|Cond|QC|Qu") %>%
  step_center(all_numeric(), -all_outcomes()) %>%
  step_scale(all_numeric(), -all_outcomes()) %>%
  step_dummy(all_nominal(), -all_outcomes(), one_hot = TRUE)

ames_train_prep <- blueprint2 %>%
  prep(ames_train) %>%
  bake(ames_train)
ames_test_prep <- blueprint2 %>%
  prep(ames_test) %>%
  bake(ames_test)

ames_dt2 <- rpart(formula = Sale_Price ~.,
                  data = ames_train_prep,
                  method = "anova",
                  control = list(cp=0, minbucket = 5, maxdepth = 3))
ames_dt2  
preds <- predict(ames_dt2, new_data = ames_test)

rpart.plot(ames_dt2)
plotcp(ames_dt2)

library(rattle)
library(RColorBrewer)

fancyRpartPlot(ames_dt2, 
               uniform = TRUE,
               palettes = "YlGnBu",
               sub = "Early stopping and Tree pruned",
               main = "Decision Tree")
?fancyRpartPlot
