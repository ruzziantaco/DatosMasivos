val list = List(2,3,4,5)

def af(list:List[Int]): Int={
var res=0
for(n <- list){
if(n==7){
res = res + 14
}
else{
res = res + n
}
}
return res
}

af(list)

val list = List(7,7,1)

af(list)