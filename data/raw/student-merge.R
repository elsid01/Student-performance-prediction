# d1=read.table("student-mat.csv",sep=";",header=TRUE)
# d2=read.table("student-por.csv",sep=";",header=TRUE)
# 
# d3=merge(d1,d2,by=c("school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"))
# print(nrow(d3)) # 382 students
# print(d3)
# names(d3)

## Load the dataset
d1 <- read.table("student-mat.csv", sep=";", header=TRUE)

cat("Rows (students):", nrow(d1), "\n")
cat("Total columns:", ncol(d1), "\n")

## Choose the target Output
y <- d1$G3
print(y)

## Define the input features X (Remove the target column)
X <- d1[, names(d1) != "G3"]
cat("Number of ML features (excluding target G3):", ncol(X), "\n")

## See Feature Names
names(X)

## Remove G1 and G2:

X <- d1[, !(names(d1) %in% c("G1","G2","G3"))]
y <- d1$G3
cat("Features excluding G1,G2:", ncol(X), "\n")




