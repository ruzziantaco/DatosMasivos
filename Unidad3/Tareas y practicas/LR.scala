//Importamos librerias necesarias
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.Pipeline
import org.apache.log4j._
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
//Eliminar errores
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
//Importamos los datos en un dataframe
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")
//Imprimimos el esquema de los datos
data.printSchema()
// Imprima un renglon de ejemplo
data.head(1)

data.select("Clicked on Ad").show()
val timedata = data.withColumn("Hour",hour(data("Timestamp")))
val logregdataall = timedata.select(data("Clicked on Ad").as("label"),$"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Hour",$"Male")
//val logregdataall = data.select(data("Clicked on Ad").as("label"),$"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Male"$"Timestamp").cast(DateType).as("Hour")
val feature_data = data.select($"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Timestamp",$"Male")
val logregdataal = (data.withColumn("Hour",hour(data("Timestamp")))
val logregdataal = logregdataall.na.drop()

val assembler = new VectorAssembler().setInputCols(Array("Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Hour","Male")).setOutputCol("features")
//val assembler = new VectorAssembler().setInputCols(Array("Area Income","Daily Internet Usage","Timestamp","Male")).setOutputCol("features")
//Utilizamos split para dividir los datos 70/30
val Array(training, test) = logregdataall.randomSplit(Array(0.7, 0.3), seed = 12345)
val lr = new LogisticRegression()

// val pipeline = new Pipeline().setStages(Array(genderIndexer,embarkIndexer,embarkEncoder,assembler,lr))
//Generamos un pipeline para entrenar los datos
val pipeline = new Pipeline().setStages(Array(assembler,lr))
//Entrenamos el modelo
val model = pipeline.fit(training)
//Transformamos el modelo con las pruebas
val results = model.transform(test)

val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

// IMprimimos la matriz de confusion
println("Confusion matrix:")
println(metrics.confusionMatrix)
//Exactitud de los datos
metrics.accuracy
