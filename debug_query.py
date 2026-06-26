import os
import time
import duckdb

print("PID:", os.getpid(), flush=True)
time.sleep(20)

con = duckdb.connect()
con.execute("CREATE TABLE t AS SELECT range AS x FROM range(1000000)")
print(con.execute("SELECT sum(x) FROM t WHERE x > 10").fetchall())
