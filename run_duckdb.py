import os
import time
import duckdb

print(f'PID: {os.getpid()}', flush=True)
print('Attach with gdb -p', os.getpid(), flush=True)

time.sleep(20)

con = duckdb.connect()
con.execute('CREATE TABLE t AS SELECT range AS x FROm range(1000000)')
print(con.execute("SELECT sum(x) FROM t WHERE x > 10"))
