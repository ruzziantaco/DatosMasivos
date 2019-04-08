

//TODO ***************************************************************************************************************************************
//******Librerias
//import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline

//Cargamos el archivo CSV como un DataFrame
val df = spark.read.option("header","true").option("inferSchema", "true")csv("/home/russo/DatosMasivos/Unidad2/Evaluacion/Iris.csv")
val struct = StructType(StructField("c0", DoubleType, true) ::StructField("c1", DoubleType, true) ::StructField("c2", DoubleType, true) ::StructField("c3",DoubleType, true) ::StructField("iris", StringType, true) :: Nil)
//creamos el nuevo dataframe esta vez con nombres en las columnas
val dfstruct = spark.read.option("header", "false").schema(struct)csv("/home/russo/DatosMasivos/Unidad2/Evaluacion/Iris.csv")
val label = new StringIndexer().setInputCol("iris").setOutputCol("label")
val assembler = new VectorAssembler().setInputCols(Array("c0", "c1", "c2", "c3")).setOutputCol("features")

//separamos los datos en dos grupos
//el de entrenamiento y el de prueba
val splits = dfstruct.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

//Especificamos las capas de nuestra red neuronal
//4 de entrada, dos capas internas, una de 5, otra de 4 neuronas y 3 de salida
val layers = Array[Int](4, 5, 4, 3)

//Creamos el entrenador y especificamos los parametros
//.setLayers es para cargar las capas de nuestra red neuronal
//.setMaxIter es para indicar el numero maximo de iteraciones 
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setLabelCol("label").setFeaturesCol("features").setPredictionCol("prediction").setBlockSize(128).setSeed(1234L).setMaxIter(100)
//creamos la tuberia de la informacion que queremos la tag y las features 
//asi como nuestro trainer que contiene las especificaciones del modelo
val pipe = new Pipeline().setStages(Array(label,assembler,trainer))
//entrenamos el modelo con los datos de nuestra tabla
val model = pipe.fit(train)


//Calculamos la exactitud en el conjunto test
val result = model.transform(test)
result.show()
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
//Imprimimos los resultados de exactitud
println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")