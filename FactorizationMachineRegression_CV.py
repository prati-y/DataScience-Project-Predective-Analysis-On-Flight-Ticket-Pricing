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

# MAGIC %md
# MAGIC ### Run the code in PySpark CLI
# MAGIC 1. Set the following to True:
# MAGIC ```
# MAGIC PYSPARK_CLI = True
# MAGIC ```
# MAGIC 1. You need to generate py (Python) file: File > Export > Source File
# MAGIC 1. Run it at your Hadoop/Spark cluster:
# MAGIC ```
# MAGIC $ spark-submit Python_Regression_Cross_Validation.py
# MAGIC ```

# COMMAND ----------

IS_DB = True # Run the code in Databricks

PYSPARK_CLI = False
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# DataFrame Schema, that should be a Table schema by Jongwook Woo (jwoo5@calstatela.edu) 01/07/2016
'''flightSchema = StructType([
  StructField("DayofMonth", IntegerType(), False),
  StructField("DayOfWeek", IntegerType(), False),
  StructField("Carrier", StringType(), False),
  StructField("OriginAirportID", IntegerType(), False),
  StructField("DestAirportID", IntegerType(), False),
  StructField("DepDelay", IntegerType(), False),
  StructField("ArrDelay", IntegerType(), False),
])'''

# COMMAND ----------

# File location and type
file_location = "/user/pyadav/Flight_Dataset_Filtered.csv"
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
  
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a temporary view of the dataframe df

# COMMAND ----------

# Create a view or table
temp_table_name = "flights_csv"
df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

'''if PYSPARK_CLI:
    csv = spark.read.csv('Flight_Dataset_Filtered_Sample-2.csv', inferSchema=True, header=True)
else:
    csv = spark.sql("SELECT * FROM flights_csv")


csv.show(5)'''

# COMMAND ----------

#data = df4.select(col("month").cast("Int").alias("Flight_Month"),col("day").cast("Int").alias("Flight_day"), "startingAirport", "destinationAirport", "fareBasisCode","isBasicEconomy", "isRefundable", "seatsRemaining","totalTravelDistance","segmentsDepartureTimeEpochSeconds", "segmentsAirlineCode", "segmentsEquipmentDescription", "segmentsDurationInSeconds", "segmentsCabinCode", "segmentsArrivalTimeEpochSeconds","baseFare")
# data = csv
#data.show(10)
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

#"startingAirport", "destinationAirport", "fareBasisCode","isBasicEconomy", "isRefundable", "seatsRemaining","totalTravelDistance","segmentsDepartureTimeEpochSeconds", "segmentsAirlineCode", "segmentsEquipmentDescription", "segmentsDurationInSeconds" , "segmentsCabinCode", "segmentsArrivalTimeEpochSeconds","baseFare")

strIdx = StringIndexer(inputCol = "startingAirport", outputCol = "startingAirportIdx")
strIdx2 = StringIndexer(inputCol = "destinationAirport", outputCol = "destinationAirportIdx")
strIdx3 = StringIndexer(inputCol = "segmentsEquipmentDescription", outputCol = "segmentsEquipmentDescriptionIdx")
strIdx4 = StringIndexer(inputCol = "segmentsAirlineCode", outputCol = "segmentsAirlineCodeIdx")
strIdx5 = StringIndexer(inputCol = "segmentsCabinCode", outputCol = "segmentsCabinCodeIdx")

#catVect = VectorAssembler(inputCols = ["isBasicEconomy", "startingAirportIdx", "destinationAirportIdx", "fareBasisCodeIdx", "segmentsAirlineCodeIdx","segmentsCabinCodeIdx"], outputCol="catFeatures")
catVect = VectorAssembler(inputCols = ["isBasicEconomy", "startingAirportIdx", "destinationAirportIdx", "segmentsAirlineCodeIdx", "segmentsCabinCodeIdx"], outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures")

# COMMAND ----------

#normalize segmentsDepartureTimeEpochSeconds, segmentsDurationInSeconds, seatsRemaining, totalTravelDistance
numVect = VectorAssembler(inputCols = ["segmentsDepartureTimeEpochSeconds", "segmentsDurationInSeconds","seatsRemaining", "totalTravelDistance", "segmentsArrivalTimeEpochSeconds"], outputCol="numFeatures")
# number vector is normalized
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

#featVect = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures", "normFeatures2", "normFeatures3", "normFeatures4", "normFeatures5" ],outputCol="features")
featVect = VectorAssembler(inputCols=["idxCatFeatures","normFeatures"],outputCol="features")


# COMMAND ----------

#fm = FMRegressor(featuresCol="scaled_features", labelCol=label_col)
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

import time
# Get the current time
current_time = time.strftime("%Y-%m-%d %H:%M:%S")
print("Time Start:", current_time)

ctv = CrossValidator(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=3)

#tv = TrainValidationSplit(estimator=pipeline, evaluator=evaluator, estimatorParamMaps=paramGrid, trainRatio=0.8)
model = ctv.fit(train)

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

fm_cvevaluator = RegressionEvaluator(predictionCol="prediction",labelCol="baseFare",metricName="r2")

print("R Squared (R2) on test data = %g" %fm_cvevaluator.evaluate(prediction))

fmv_cvevaluator = RegressionEvaluator(labelCol="baseFare", predictionCol="prediction", metricName="rmse")

print("RMSE: %f" % fmv_cvevaluator.evaluate(prediction))
