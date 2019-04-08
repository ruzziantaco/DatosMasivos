val list = List(2,3,4,5)

def ParLista(list:List[Int]): String ={
for(n <- list){
if(n%2==0){
println(s"$n es par")
}
else{
println(s"$n es inpar")
}
}
return "Done"
}

ParLista(list)