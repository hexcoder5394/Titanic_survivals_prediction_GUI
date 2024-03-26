import mysql.connector
import pandas as pd
mydb = mysql.connector.connect(
    host = "localhost",
    database = "titanicDB",
    user = "achintha",
    password = "5394"
)

query = "select PassengerId, Name from train;"
data_frame = pd.read_sql(query, mydb)
print(data_frame)
mydb.close()

