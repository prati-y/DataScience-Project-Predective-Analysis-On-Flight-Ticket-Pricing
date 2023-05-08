# Databricks notebook source
# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression, FMRegressor, RandomForestRegressor, GBTRegressionModel, GBTRegressor
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.regression import LinearRegression, GBTRegressor, RandomForestRegressor, FMRegressor
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import FMRegressor

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


# COMMAND ----------

IS_DB = True # Run the code in Databricks

PYSPARK_CLI = False
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Flight_Dataset_Filtered_sample.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
  
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a temporary view of the dataframe df

# COMMAND ----------

# Create a view or table
temp_table_name = "flights_csv"
df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

data = df.select("startingAirport", "destinationAirport", "fareBasisCode","isBasicEconomy", "isRefundable", "seatsRemaining","totalTravelDistance","segmentsDepartureTimeEpochSeconds", "segmentsArrivalTimeEpochSeconds" ,"segmentsAirlineCode", "segmentsEquipmentDescription", "segmentsDurationInSeconds" , "segmentsCabinCode", "baseFare")
data = data.withColumn("isBasicEconomy", col("isBasicEconomy").cast("int"))
data.printSchema()
data.show()


# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1]
train_rows = train.count()
test_rows = test.count()
print("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

data.printSchema()

# COMMAND ----------

data.head()

# COMMAND ----------

# DBTITLE 1,Define the Pipeline
strIdx = StringIndexer(inputCol = "startingAirport", outputCol = "startingAirportIdx")
strIdx2 = StringIndexer(inputCol = "destinationAirport", outputCol = "destinationAirportIdx")
strIdx3 = StringIndexer(inputCol = "segmentsEquipmentDescription", outputCol = "segmentsEquipmentDescriptionIdx")
strIdx4 = StringIndexer(inputCol = "segmentsAirlineCode", outputCol = "segmentsAirlineCodeIdx")
strIdx5 = StringIndexer(inputCol = "segmentsCabinCode", outputCol = "segmentsCabinCodeIdx")

catVect = VectorAssembler(inputCols = ["isBasicEconomy", "startingAirportIdx", "destinationAirportIdx", "segmentsAirlineCodeIdx", "segmentsCabinCodeIdx"], outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures")

# COMMAND ----------

numVect = VectorAssembler(inputCols = ["segmentsDepartureTimeEpochSeconds", "segmentsDurationInSeconds","seatsRemaining", "totalTravelDistance", "segmentsArrivalTimeEpochSeconds"], outputCol="numFeatures")

# number vector is normalized
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

featVect = VectorAssembler(inputCols=["idxCatFeatures","normFeatures"],outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient Boosted Tree

# COMMAND ----------

paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [8, 12]) \
    .addGrid(gbt.minInfoGain, [0.0]) \
    .addGrid(gbt.maxBins, [64]) \
    .addGrid(gbt.maxIter, [5]) \
    .build()

rf_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="baseFare", metricName="r2")


# COMMAND ----------

pipeline = Pipeline(stages=[strIdx, strIdx2, strIdx3, strIdx4, strIdx5,catVect, catIdx,numVect,minMax,featVect,gbt])

# COMMAND ----------

# DBTITLE 1,Train Validation
import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

tv = TrainValidationSplit(estimator=pipeline, evaluator=gbt_evaluator,estimatorParamMaps=paramGrid, trainRatio=0.8)
model = tv.fit(train)

current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time End:", current_time)

# COMMAND ----------

testing = model.transform(test).select(col("features"),col("baseFare").alias("trueLabel"))
testing.show()

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", col("baseFare").alias("trueLabel"))
predicted.show()

# COMMAND ----------

display(predicted)

# COMMAND ----------

# DBTITLE 1,Calculate the rmse and R2
#Calculatethe rmse and R2

gbttv_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %gbttv_evaluator.evaluate(prediction))

gbttv_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % gbttv_evaluator.evaluate(prediction))

# COMMAND ----------

# DBTITLE 1,Cross Validation
import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

K = 3
cv = CrossValidator(estimator=pipeline, evaluator=rf_evaluator, estimatorParamMaps=paramGrid, numFolds = K)
model = cv.fit(train)

current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time End:", current_time)

# COMMAND ----------

testing = model.transform(test).select(col("features"),col("baseFare").alias("trueLabel"))
testing.show()

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", col("baseFare").alias("trueLabel"))
predicted.show()

# COMMAND ----------

# DBTITLE 1,Calculate the rmse and R2
#Calculate the rmse and R2

gbtcv_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %gbtcv_evaluator.evaluate(prediction))

gbtcv_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % gbtcv_evaluator.evaluate(prediction))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Regression

# COMMAND ----------

rf= RandomForestRegressor(labelCol="baseFare", featuresCol="features")

paramGrid = ParamGridBuilder() \
.addGrid(rf.maxDepth, [13, 16]) \
.addGrid(rf.minInfoGain, [0.0]) \
.addGrid(rf.maxBins, [64]) \
.build()

rf_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="baseFare", metricName="r2")

# COMMAND ----------

pipeline = Pipeline(stages=[strIdx, strIdx2, strIdx3, strIdx4, strIdx5,catVect, catIdx, numVect, minMax, featVect,rf])
print ("Pipeline complete!")

# COMMAND ----------

# DBTITLE 1,Train Validation
import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

tv = TrainValidationSplit(estimator=pipeline, evaluator=rf_evaluator,estimatorParamMaps=paramGrid, trainRatio=0.8)
model = tv.fit(train)

current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time End:", current_time)

testing = model.transform(test).select(col("features"),col("baseFare").alias("trueLabel"))
testing.show()

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", col("baseFare").alias("trueLabel"))
predicted.show()

display(predicted)


# COMMAND ----------

# DBTITLE 1,Calculate the rmse and R2
#Calculatethe rmse and R2

rftv_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %rftv_evaluator.evaluate(prediction))

rftv_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % rftv_evaluator.evaluate(prediction))

# COMMAND ----------

# DBTITLE 1,Cross Validation
import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

K = 3
cv = CrossValidator(estimator=pipeline, evaluator=rf_evaluator, estimatorParamMaps=paramGrid, numFolds = K)
model = cv.fit(train)

current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time End:", current_time)

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", col("baseFare").alias("trueLabel"))
predicted.show()

# COMMAND ----------

display(predicted)

# COMMAND ----------

# DBTITLE 1,Calculate the rmse and R2
rfcv_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %rfcv_evaluator.evaluate(prediction))

rfcv_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % rfcv_evaluator.evaluate(prediction))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Factorization Machines Regressor

# COMMAND ----------

fm = FMRegressor(labelCol="baseFare", featuresCol="features")

# COMMAND ----------

paramGrid = ParamGridBuilder() \
     .addGrid(fm.stepSize, [1, 0.5]) \
     .build()


# COMMAND ----------

evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="r2")


# COMMAND ----------

pipeline = Pipeline(stages=[strIdx, strIdx2, strIdx3, strIdx4, strIdx5, catVect, catIdx, numVect, minMax, featVect, fm])


# COMMAND ----------

# DBTITLE 1,Train Validation
import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

#tv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)

tv = TrainValidationSplit(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, trainRatio=0.8)
model = tv.fit(train)

current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time End:", current_time)

# COMMAND ----------

testing = model.transform(test).select(col("features"),col("baseFare").alias("trueLabel"))
testing.show()

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", col("baseFare").alias("trueLabel"))
predicted.show()

# COMMAND ----------

display(predicted)

# COMMAND ----------

# DBTITLE 1,Calculate the rmse and R2
fm_tvevaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %fm_tvevaluator.evaluate(prediction))

fmv_tvevaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % fmv_tvevaluator.evaluate(prediction))

# COMMAND ----------

# DBTITLE 1,Cross Validation
import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

cv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)
model = cv.fit(train)

current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time End:", current_time)

# COMMAND ----------

testing = model.transform(test).select(col("features"),col("baseFare").alias("trueLabel"))
testing.show()

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", col("baseFare").alias("trueLabel"))
predicted.show()

# COMMAND ----------

# DBTITLE 1,Calculate the rmse and R2
fm_cvevaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %fm_cvevaluator.evaluate(prediction))

fm_cvevaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % fm_cvevaluator.evaluate(prediction))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Decision Tree Regression

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(labelCol = "baseFare", featuresCol="features" , maxBins= 3000)

paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [6,9]) \
    .addGrid(dt.minInfoGain, [0.0]) \
    .addGrid(dt.maxBins, [65,50,65]) \
    .build()

dt_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="baseFare", metricName="r2")

pipeline = Pipeline(stages=[strIdx, strIdx2, strIdx3, strIdx4, strIdx5,catVect, catIdx, numVect, minMax, featVect,dt])
print ("Pipeline complete!")

# COMMAND ----------

# DBTITLE 1,Train Validation
import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

tv = TrainValidationSplit(estimator=pipeline, evaluator=dt_evaluator,estimatorParamMaps=paramGrid, trainRatio=0.8)
model = tv.fit(train)

current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time End:", current_time)


# COMMAND ----------

testing = model.transform(test).select(col("features"),col("baseFare").alias("trueLabel"))
testing.show()


# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", col("baseFare").alias("trueLabel"))
predicted.show()

# COMMAND ----------

# DBTITLE 1,Calculate the rmse and R2
#Calculatethe rmse and R2

dttv_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %dttv_evaluator.evaluate(prediction))

dttv_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % dttv_evaluator.evaluate(prediction))


# COMMAND ----------

# DBTITLE 1,Cross Validation
import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

K = 3
cv = CrossValidator(estimator=pipeline, evaluator=dt_evaluator, estimatorParamMaps=paramGrid, numFolds = K)

model = cv.fit(train)

current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time End:", current_time)

# COMMAND ----------

testing = model.transform(test).select(col("features"),col("baseFare").alias("trueLabel"))
testing.show()

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", col("baseFare").alias("trueLabel"))
predicted.show()

# COMMAND ----------

# DBTITLE 1,Calculate the rmse and R2
#Calculatethe rmse and R2

dtcv_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %dtcv_evaluator.evaluate(prediction))

dtcv_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % dtcv_evaluator.evaluate(prediction))

