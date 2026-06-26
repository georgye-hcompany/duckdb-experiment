import os
import time
import duckdb

print("PID:", os.getpid(), flush=True)
print("Attach with: gdb -p", os.getpid(), flush=True)

# Gives you time to attach.
time.sleep(30)

con = duckdb.connect()
con.execute("CREATE TABLE t AS SELECT range AS x FROM range(1000000)")
print(con.execute("SELECT sum(x) FROM t WHERE x > 10").fetchall())