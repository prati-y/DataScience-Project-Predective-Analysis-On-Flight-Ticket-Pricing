# Databricks notebook source
# MAGIC %md
# MAGIC ## Overview
# MAGIC
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# File location and type
#file_location = "/FileStore/tables/Flight_Dataset_Filtered_Sample.csv"
#file_location = "/user/pwong4/Flight_Dataset_Filtered_Sample.csv"
file_location = "/user/pwong4/Flight_Dataset_Filtered.csv"
file_type = "csv"

# CSV options
infer_schema = "True"
first_row_is_header = "True"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "Flight_Dataset_Filtered_Sample_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC
# MAGIC select * from `Flight_Dataset_Filtered_Sample_csv`

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "Flight_Dataset_Filtered_Sample_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# DBTITLE 1,Prepare the data
data = df.select("startingAirport", "destinationAirport", "fareBasisCode","isBasicEconomy", "isRefundable", "seatsRemaining","totalTravelDistance","segmentsDepartureTimeEpochSeconds", "segmentsArrivalTimeEpochSeconds" ,"segmentsAirlineCode", "segmentsEquipmentDescription", "segmentsDurationInSeconds" , "segmentsCabinCode", "baseFare")
data = data.withColumn("isBasicEconomy", col("isBasicEconomy").cast("int"))
data.printSchema()
data.show()

# COMMAND ----------

# DBTITLE 1,Split the Data
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

catVect = VectorAssembler(inputCols = ["isBasicEconomy", "startingAirportIdx", "destinationAirportIdx", "segmentsAirlineCodeIdx","segmentsCabinCodeIdx", "segmentsEquipmentDescriptionIdx"], outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures")


# COMMAND ----------


#normalize segmentsDepartureTimeEpochSeconds, segmentsDurationInSeconds, seatsRemaining, totalTravelDistance
numVect = VectorAssembler(inputCols = ["segmentsDepartureTimeEpochSeconds", "segmentsDurationInSeconds","seatsRemaining", "totalTravelDistance", "segmentsArrivalTimeEpochSeconds"], outputCol="numFeatures")
# number vector is normalized
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

#featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures", "normFeatures2", "normFeatures3", "normFeatures4", "normFeatures5" ],outputCol="features")
featVect = VectorAssembler(inputCols=["idxCatFeatures","normFeatures"],outputCol="features")


# COMMAND ----------

# DBTITLE 1,Train a Regression Model 
#gbt = GBTRegressor(labelCol="baseFare", featuresCol="features")
gbt = GBTRegressor(labelCol="baseFare", featuresCol="features")
#paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth, [2, 5]).addGrid(gbt.maxIter, [2, 5]).build()

'''
r662 sample, r.684
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [6,8,10]) \
    .addGrid(gbt.minInfoGain, [0.0]) \
    .addGrid(gbt.maxBins, [64]) \
    .addGrid(gbt.maxIter, [5]) \
    .build()
'''
''' BEST FULL r.7061 rmse.83.17 w equipmentdescription  Sample was .62
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [12]) \
    .addGrid(gbt.minInfoGain, [0.0]) \
    .addGrid(gbt.maxBins, [64]) \
    .addGrid(gbt.maxIter, [5]) \
    .build()
'''
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [12]) \
    .addGrid(gbt.minInfoGain, [0.0]) \
    .addGrid(gbt.maxBins, [64]) \
    .addGrid(gbt.maxIter, [5]) \
    .build()
    
'''
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [13, 16]) \
    .build()
'''
 #   .addGrid(gbt.stepSize, [0.01, 0.001]) \
gbt_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="baseFare", metricName="r2")

                    
#rf = RandomForestRegressor(labelCol="baseFare",featuresCol="features", numTrees=10, maxDepth=5)

#dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")
#dt = LogisticRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=0.3)
#lr = LinearRegression(labelCol="baseFare",featuresCol="features",maxIter=10, regParam=0.3)



# COMMAND ----------

# Pipeline process the series of transformation above, which is 7 transformation
#pipeline = Pipeline(stages=[strIdx, strIdx2, strIdx3, strIdx4, strIdx5, catVect, catIdx, numVect, minMaxEpoch, minMaxDuration, featVect, gbt]) #minMax2, dt])
pipeline = Pipeline(stages=[strIdx, strIdx2, strIdx3, strIdx4, strIdx5,catVect, catIdx,numVect,minMax,featVect,gbt])


#piplineModel = pipeline.fit(train)
print ("Pipeline complete!")

# COMMAND ----------

# DBTITLE 1,Train Validation Split
import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

tv = TrainValidationSplit(estimator=pipeline, evaluator=gbt_evaluator,estimatorParamMaps=paramGrid, trainRatio=0.8)
model = tv.fit(train)

current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time End:", current_time)

# COMMAND ----------
#feature importance does not work CV or TV so need to fit with pipeline ie pipeline.fit(train)
'''
import pandas as pd

gbtModel = model.stages[-1]
print(gbtModel.toDebugString)


featureImp = pd.DataFrame(list(zip(featVect.getInputCols(), gbtModel.featureImportances)),
columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending=False)

#for Cat or Numerical
featureImp2 = pd.DataFrame(list(zip(catVect.getInputCols(), gbtModel.featureImportances)),
columns=["feature", "importance"])
#featureImp2.sort_values(by="importance", ascending=False)

featureImp3 = pd.DataFrame(list(zip(numVect.getInputCols(), gbtModel.featureImportances)),
columns=["feature", "importance"])
#featureImp3.sort_values(by="importance", ascending=False)

featureImp = pd.concat([featureImp2, featureImp3], axis=0)
featureImp = featureImp.sort_values(by="importance", ascending=False)
print(featureImp)
'''
# COMMAND ----------

# DBTITLE 1,Test the Model
testing = model.transform(test).select(col("features"),col("baseFare").alias("trueLabel"))
testing.show()

# COMMAND ----------

prediction = model.transform(test)
predicted = prediction.select("features", "prediction", col("baseFare").alias("trueLabel"))
predicted.show()

# COMMAND ----------

display(predicted)

# COMMAND ----------

# DBTITLE 1,Calculate the RMSE and R2
#Calculatethe rmse and R2

gbttv_evaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %gbttv_evaluator.evaluate(prediction))

gbttv_evaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % gbttv_evaluator.evaluate(prediction))
