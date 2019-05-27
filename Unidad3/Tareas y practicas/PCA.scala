//Importamos las librerias de spark necesarias
import org.apache.spark.ml.feature.{PCA,StandardScaler,VectorAssembler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
//Cargamos e un dataframe los datos del csv
val spark = SparkSession.builder().appName("PCA_Example").getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Cancer_Data")
//Vemos el esquema de los datos
data.printSchema()

//Colocamos en un arreglo las columnas
val columnas = (Array("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
"mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
"radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error",
"concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius",
"worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity",
"worst concave points", "worst symmetry", "worst fractal dimension"))
//Se le da las columnas que se llama "columnas" que vamos a entrenar
val assembler = new VectorAssembler().setInputCols(columnas).setOutputCol("features")

val output = assembler.transform(data).select($"features")


//Colocamos los datos de entrada para luego hacer fit dentro del modelo
val scaler = (new StandardScaler()
 .setInputCol("features")
 .setOutputCol("scaledFeatures")
 .setWithStd(true)
 .setWithMean(false))
val scalerModel = scaler.fit(output)


//Transformamos el modelo para poder especificar los parametros de PCA
val scaledData = scalerModel.transform(output)

//Especificamos los parametros del PCA
val pca = (new PCA()
 .setInputCol("scaledFeatures")
 .setOutputCol("pcaFeatures")
 .setK(4)
 .fit(scaledData))

val pcaDF = pca.transform(scaledData)
//Se imprime el resultado
val result = pcaDF.select("pcaFeatures")
result.show()

result.head(1)
