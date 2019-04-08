//1. Desarrollar un algoritmo en scala que calcule el radio de un círculo
scala> def radio(Area:Double):Double={ 
     |(math.sqrt(Area/3.1416))
     |}
radio: (Area: Double)Double

scala> radio(78.54)
res12: Double = 5.0


//2. Desarrollar un algoritmo en scala que me diga si un número es primo
scala> def isPrime(num:Int):Boolean =
   |(num > 1) && !(2 to scala.math.sqrt(num).toInt).exists(x =>num % x == 0)
isPrime: (num: Int)Boolean

scala> isPrime(15)
res16: Boolean = false

scala> isPrime(3)
res17: Boolean = true


//3. Dada la variable bird = "tweet", utiliza interpolación de string para
//   imprimir "Estoy escribiendo un tweet"

scala> var bird = "tweet"
bird: String = tweet

scala> println(s"Estoy escribiendo un $bird")
Estoy escribiendo un tweet


//4. Dada la variable mensaje = "Hola Luke yo soy tu padre!" utiliza slice para extraer la
//   secuencia "Luke"
scala> var mensaje = "Hola Luke yo soy tu padre!"
mensaje: String = Hola Luke yo soy tu padre!

scala> mensaje
res6: String = Hola Luke yo soy tu padre!

scala> mensaje.slice(5,9)
res7: String = Luke

scala> 



//5. Cual es la diferencia en value y una variable en scala?
scala>  val Num1=1
Num1: Int = 1

scala> Num1=2
<console>:25: error: reassignment to val
       Num1=2
           ^
scala> var Num2=2
Num2: Int = 2

scala> Num2=3
Num2: Int = 3



//6. Dada la tupla ((2,4,5),(1,2,3),(3.1416,23))) regresa el número 3.1416
scala> val Tupla=((2,4,5),(1,2,3),(3.1416,23))
Tupla: ((Int, Int, Int), (Int, Int, Int), (Double, Int)) = ((2,4,5),(1,2,3),(3.1416,23))

scala> Tupla
res23: ((Int, Int, Int), (Int, Int, Int), (Double, Int)) = ((2,4,5),(1,2,3),(3.1416,23))

scala> println(Tupla._3._1)
3.1416


