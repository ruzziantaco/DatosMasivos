def balance(list:List[Int]): Boolean={
var primera=0
var segunda=0
segunda=list.sum
for(i<-Range(0,list.length)){
primera=primera+list(i)
segunda=segunda-list(i)
if(primera==segunda){
return true
}
}
return false
}

val bl = List(3,2,1)
val bl2 = List(2,3,3,2)
val bl3 = List(10,30,90)

balance(bl)
balance(bl2)
balance(bl3)