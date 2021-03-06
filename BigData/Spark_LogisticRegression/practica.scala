import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics


Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val df  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")

df.printSchema()

df.head(1)

df.show(1)

val datos = (df.select(df("Clicked on Ad").as("label"), $"Daily Time Spent on Site", $"Age", $"Area Income", $"Daily Internet Usage", df("Timestamp").as("Hour"), $"Male"))
datos.show(1)

import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

val assembler = (new VectorAssembler().setInputCols(Array("Daily Time Spent on Site","Age", "Area Income","Daily Internet Usage","Hour")).setOutputCol("features"))
val Array(training, test) = datos.randomSplit(Array(0.7, 0.3), seed = 12345)


val lr = new LogisticRegression()

val pipeline = new Pipeline().setStages(Array(assembler,lr))

val model = pipeline.fit(training)

val results = model.transform(test)

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// Matriz de confusion
println("Confusion matrix:")
println(metrics.confusionMatrix)

metrics.accuracy
