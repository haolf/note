/*基础格式*/
WITH new_table AS (),
  new_table2 AS ()           /*CTE*/

SELECT orders_1.a,           /*选定表的指定a,*时表示所有列*/
FROM orders_1                /*选定指定表orders_1*/
JOIN orders_2                /*加入表格orders_2*/
ON orders_1.a = orders_2.A   /*指定两个表格之间的某个列的联系*/
WHERE a = 11                 /*选定a列中值为11的行,或使用其他运算符/逻辑运算符*/
GROUP BY a                   /*用来在数据子集中聚合数据,*/
HAVING SUM(x) > 1            /*用于聚合列的where查询*/
ORDER BY a                   /*以a列升序排列所有行*/
LIMIT 10                     /*查看指定的前10行*/

/*拓展示例*/
SELECT COUNT(*)         /*COUNT函数会返回指定列所有包含非空值数据行的计数*/
SELECT SUM(a)           /*SUM函数返回指定数值列的总和,空值在SUM中被当为0处理*/
SELECT MAX()            /*返回最小的数字、最早的日期或按字母表排序的最之前的非数字值*/
SELECT MIN()            /*与MAX相反,MIN 和 MAX 聚合函数也会忽略 NULL 值*/
SELECT AVG()            /*返回的是数据的平均值,空值会被忽略*/
SELECT DISTINCT         /*仅返回唯一值*/
SELECT a + b AS d       /*指定一个值为a列+b列的新列,并命名为d,(AS可省略)*/
WHERE a LIKE '%A%'      /*选定a列中所有值包含A的行*/
WHERE a IN ('A','B')    /*选定a列中所有值为A或B的行*/
WHERE a IS NULL         /*找到a列中值为空值的行*/
WHERE a >= 6 AND a <= 10/*选定a列中值大于等于A并且小于等于B的行*/
WHERE a BETWEEN 6 AND 10/*同上*/
ORDER BY c DESC         /*以c列降序排列所有行*/
ORDER BY a DESC, c      /*升序排列a列,为a列的每个唯一值降序排列c列*/
SELECT COALESCE(a,'x')  /*在新列中用x填充a列中的空值*/
SELECT LEFT(a,1) AS qian/*从左侧开始，从特定列a中的每行选取一定数量(1)的字符*/
SELECT RIGHT(a,1) AS hou/*从右侧开始，从特定列a中的每行选取一定数量(1)的字符*/
SELECT POSITION('A' IN a) AS xiao/*获取a列中字符A在每个值中的位置(数字1开始计数)*/
SELECT STRPOS(a,'A') AS xiao     /*同上,格式不同,均区分大小写*/
SELECT SUBSTR(a,1,5)    /*提取a列中指定字符串,自1开始共5个字符*/
SELECT CONCAT(a,'.',b)  /*将a列与'.'与b列的值都组合在一起*/
SELECT a || '.' || b    /*同上*/
LENGTH()                /*获取字符串字符个数*/
UPPER()                 /*大写字符串*/
LOWER()                 /*小写字符串*/
TO_DATE(a,'month')      /*将a列中的值转换数据类型为时间戳月*/
CAST(a AS date) AS dates/*将a列中的值转换为日期格式并重命名列为dates*/
(a)::date AS dates      /*同上*/
SELECT CASE WHEN a='f' THEN 'big'
            WHEN a='z' THEN 'small'
            ELSE 'null' END AS f
/*创建一个列,新列的值取决于a列中的值*/

SELECT DATE_TRUNC(‘interval’, time_column) AS x
/*使你能够将日期截取到日期时间列的特定部分*/
SELECT DATE_PART(‘interval’, time_column)
/*可以用来获取日期的特定部分*
interval:
/*
microsecond /*微秒*
millisecond /*毫秒*
second      /*秒*
minute      /*分钟*
hour        /*小时*
day         /*天*
week        /*星期*
dow         /*DATE_PART中以0到6数字返回星期*
month       /*月*
quarter     /*季度*
year        /*年*
decade      /*十年*
century     /*世纪*
millenium   /*千年*
*/

/*逻辑运算符*/
LIKE        /*用于进行类似于使用WHERE和=的运算，不知道自己想准确查找哪些内容的情况。*/
IN          /*用于执行类似于使用 WHERE 和 = 的运算，但用于多个条件的情况。*/
NOT         /*这与IN和LIKE一起使用，用于选择 NOT LIKE 或 NOT IN 某个条件的所有行*/
AND&BETWEEN /*可用于组合所有组合条件必须为真的运算。*/
OR          /*可用于组合至少一个组合条件必须为真的运算。*/
