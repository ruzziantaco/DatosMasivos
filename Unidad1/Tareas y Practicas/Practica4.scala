//------------Fibonacci

scala> def fun(n:Int):Int={
     |     if(n<2){
     |     return n
     | }
     | else{
     |     return((n -1)+(n+2))
     | }
     | }
fun: (n: Int)Int

scala> fun(8)
res1: Int = 17


scala> def fun(n:Double): Double ={
     |     if(n<2){
     |         return n
     |     }
     |     else {
     |         var m = ((1+math.sqrt(5))/2)
     |         var j = ((math.pow(m,n)-math.pow((1-m),n))/math.sqrt(5))
     |         return j
     |     }
     | }
fun: (n: Double)Double

scala> fun(9)
res1: Double = 34.00000000000001


scala> def func(x : Int):Int={
     |     var a = 0
     |     var b = 1
     |     var c =0
     |     var k = 0
     |     for(k <- 1 to x){
     |         println(a)
     |         c=b+a
     |         a=b
     |         b=c            
     |     }
     |     return(x)
     | }
func: (x: Int)Int

scala> func(8)
0
1
1
2
3
5
8
13
res11: Int = 8



scala> def func(x : Int):Int={
     |     var a = 0
     |     var b = 1
     |     var k = 0
     |     for(k <- 1 to x){
     |         println(a)
     |         b=b+a
     |         a=b-a        
     |     }
     |     return(x)
     | }
func: (x: Int)Int

scala> func(8)
0
1
1
2
3
5
8
13
res16: Int = 8

