//---1
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

//2
val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv") 

//3
df.printSchema()

//4
df.printSchema()

//5
df.select($"Date", $"Open", $"High",$"Low",$"Close").show()

//6
df.describe()

//7
val df2 = df.withColumn("HV Ratio", df("High")/df("Volume"))
df2.select("HV Ratio").show()

//8
df.select(max("High")).show()
df.groupBy($"High".desc).show()

//9
print("El csv es una muestra del precio al que estaban las acciones en la bolsa de valores de la compania Netllix, la columna Close representa el precio con el que cerro el valor por accion de Netflix de ese dia.")

//10
df.select(max("Volume")).show()
df.select(min("Volume")).show()

//11
//a 
df.filter($"Close"<600).count()

//b
print((df.filter($"High">500).count()*100)/1250,"%")

//c
df.select(corr("High", "Volume")).show()

//d
val df2=df.withColumn("Year",year(df("Date")))
val dfmax=df2.groupBy("Year").max()
dfmax.select($"Year",$"max(High)").show()
dfmax.select($"Year",$"max(High)").show(1)

//e
val dfmonth=df.withColumn("Month",month(df("Date")))
val dfmean=dfmonth.select($"Month",$"Close").groupBy("Month").mean()
dfmean.orderBy($"Month".desc).show()
dfmean.orderBy($"Month").show()

