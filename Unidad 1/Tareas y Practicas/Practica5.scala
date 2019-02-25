// 1 - computes sqrt(a2 + b2) without intermediate overflow or underflow.
scala> df.select(hypot(column("Sales"),column("Sales"))).show()
+-------------------+
|HYPOT(Sales, Sales)|
+-------------------+
|   282.842712474619|
|  169.7056274847714|
| 480.83261120685233|
|  848.5281374238571|
| 175.36248173426378|
|  343.6538956566621|
| 1230.3657992645926|
|  494.9747468305833|
+-------------------+

// 2 - Returns a reversed string or an array with reverse order of elements.
scala> df.select(reverse(column("Person"))).show()
+---------------+
|reverse(Person)|
+---------------+
|            maS|
|        eilrahC|
|          knarF|
|           aniT|
|            ymA|
|        assenaV|
|           lraC|
|          haraS|
+---------------+

// 3 - An expression that returns the string representation of the binary value of the given long column.
scala> df.select(bin("Sales")).show()
+----------+
|bin(Sales)|
+----------+
|  11001000|
|   1111000|
| 101010100|
|1001011000|
|   1111100|
|  11110011|
|1101100110|
| 101011110|
+----------+

// 4 - Computes the Levenshtein distance of the two given string columns.
scala> df.select(levenshtein(column("Company"),column("Person"))).show()
+----------------------------+
|levenshtein(Company, Person)|
+----------------------------+
|                           4|
|                           7|
|                           5|
|                           4|
|                           4|
|                           7|
|                           4|
|                           5|
+----------------------------+

// 5 - Returns the value of the first argument raised to the power of the second argument.
scala> df.select(pow("Sales",2)).show()
+-----------------+
|POWER(Sales, 2.0)|
+-----------------+
|          40000.0|
|          14400.0|
|         115600.0|
|         360000.0|
|          15376.0|
|          59049.0|
|         756900.0|
|         122500.0|
+-----------------+

// 6 - Computes the exponential of the given column.
scala> df.select(exp(last("Sales"))).show()
+-----------------------+
|EXP(last(Sales, false))|
+-----------------------+
|   1.007090887028079...|
+-----------------------+

// 7 - Computes the cube-root of the given column.
scala> df.select(cbrt("Sales")).show()
+-----------------+
|      CBRT(Sales)|
+-----------------+
|5.848035476425732|
|4.932424148660941|
|6.979532046908887|
|8.434326653017493|
|4.986630952238646|
|6.240251469155712|
| 9.54640270936004|
|7.047298732064892|
+-----------------+

// 8 - Aggregate function: returns the sum of all values in the given column.
scala> df.select(sum("Sales")).show()
+----------+
|sum(Sales)|
+----------+
|      2847|
+----------+

// 9 - Aggregate function: returns the population standard deviation of the expression in a group.
scala> df.select(stddev_pop("Sales")).show()
+-----------------+
|stddev_pop(Sales)|
+-----------------+
| 242.551251027489|
+-----------------+

// 10 - Aggregate function: returns the average of the values in a group.
scala> df.select(avg("Sales")).show()
+----------+
|avg(Sales)|
+----------+
|   355.875|
+----------+

// 11 - Computes the character length of a given string or number of bytes of a binary string.
scala> df.select(length(last("Company"))).show()
+----------------------------+
|length(last(Company, false))|
+----------------------------+
|                           2|
+----------------------------+

// 12 - Computes the numeric value of the first character of the string column, and returns the result as an int column.
scala> df.select(ascii(last("Company"))).show()
+---------------------------+
|ascii(last(Company, false))|
+---------------------------+
|                         70|
+---------------------------+

// 13 - Converts a string column to lower case.
scala> df.select(lower(first("Company"))).show()
+----------------------------+
|lower(first(Company, false))|
+----------------------------+
|                        goog|
+----------------------------+

// 14 - Converts a string column to upper case.
scala> df.select(upper(column("Person"))).show()
+-------------+
|upper(Person)|
+-------------+
|          SAM|
|      CHARLIE|
|        FRANK|
|         TINA|
|          AMY|
|      VANESSA|
|         CARL|
|        SARAH|
+-------------+

// 15 - Computes the tan of the specified float value.
scala>  df.select(tan(last("Sales"))).show()
+-----------------------+
|COS(last(Sales, false))|
+-----------------------+
|   -0.28363327918216646|
+-----------------------+

// 16 - Computes the tan of the specified column values.
scala> df.select(tan(column("Sales"))).show()
+--------------------+
|          TAN(Sales)|
+--------------------+
| -1.7925274837903817|
|  0.7131230097859091|
|  0.8560456341220392|
|-0.04422563560116593|
|  10.732137526072965|
|  1.9520219903034328|
|-0.22484304402728392|
|   3.380889674884478|
+--------------------+

// 17 - Aggregate function: returns the first value in a group.
df.select(first("Sales")).show()
scala> df.select(first("Sales")).show()
+-------------------+
|first(Sales, false)|
+-------------------+
|                200|
+-------------------+

// 18 - Aggregate function: returns the last value in a group.
df.select(last("Sales")).show()
scala> df.select(last("Sales")).show()
+------------------+
|last(Sales, false)|
+------------------+
|               350|
+------------------+

// 19 - Computes the square root of the specified float value.
scala> df.select(sqrt("Sales")).show()
+------------------+
|       SQRT(Sales)|
+------------------+
|14.142135623730951|
|10.954451150103322|
|18.439088914585774|
| 24.49489742783178|
|11.135528725660043|
|15.588457268119896|
| 29.49576240750525|
|18.708286933869708|
+------------------+

// 20 - Computes the sin of the specified column values.
scala> df.select(sin("Sales")).show()
+--------------------+
|          SIN(Sales)|
+--------------------+
| -0.8732972972139946|
|  0.5806111842123143|
|  0.6503107401625525|
|0.044182448331873195|
| -0.9956869868891794|
| -0.8900093488562771|
| 0.21936643855075033|
| -0.9589328250406132|
+--------------------+