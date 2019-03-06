//-----Pregunta 1
scala> def Par(num:Int): Boolean = num%2 == 0
Par: (num: Int)Boolean

scala> Par(2)
res1: Boolean = true

scala> Par(3)
res2: Boolean = false



//-----Pregunta 2

scala> val list = List(2,3,4,5)
list: List[Int] = List(2, 3, 4, 5)

scala> def ParLista(list:List[Int]): String ={
     | for(n <- list){
     | if(n%2==0){
     | println(s"$n es par")
     | }
     | else{
     | println(s"$n es inpar")
     | }
     | }
     | return "Done"
     | }
ParLista: (list: List[Int])String

scala> ParLista(list)
2 es par
3 es inpar
4 es par
5 es inpar
res0: String = Done


//-----Pregunta 3

scala> val list = List(2,3,4,5)
list: List[Int] = List(2, 3, 4, 5)

scala> def af(list:List[Int]): Int={
     | var res=0
     | for(n <- list){
     | if(n==7){
     | res = res + 14
     | }
     | else{
     | res = res + n
     | }
     | }
     | return res
     | }
af: (list: List[Int])Int

scala> af(list)
res3: Int = 14

scala> val list = List(2,3,4,5,6)
list: List[Int] = List(2, 3, 4, 5, 6)

scala> af(list)
res4: Int = 20

//-----Pregunta 4

scala> def balance(list:List[Int]): Boolean={
| var primera=0
| var segunda=0
| segunda=list.sum
| for(i<-Range(0,list.length)){
| primera=primera+list(i)
| segunda=segunda-list(i)
| if(primera==segunda){
| return true
| }
| }
| return false
| }

balance: (list: List[Int])Boolean

scala> val numeros=List(1,4,4,1)
numeros: List[Int] = List(1, 4, 4, 1)
scala> balance(numeros)
res5: Boolean = true



//-----Pregunta 5

scala> def palindromo(palabra:String):Boolean ={
     | return (palabra == palabra.reverse)
     | }
palindromo: (palabra: String)Boolean

scala> var palabra = "ana"
palabra: String = ana

scala> palindromo(palabra)
res0: Boolean = true

scala> var palabra = "anx"
palabra: String = anx

scala> palindromo(palabra)
res1: Boolean = false