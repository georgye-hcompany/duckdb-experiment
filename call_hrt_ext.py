import os
import time
import duckdb

EXT = "/home/ubuntu/duckdb-gdb-playground/hrt_duck_ext/build/release/repository/v1.5.2/linux_amd64/hrt_ext.duckdb_extension"

print("PID:", os.getpid(), flush=True)
print("DuckDB Python version:", duckdb.__version__, flush=True)
print("Extension:", EXT, flush=True)
print("Attach with: gdb -p", os.getpid(), flush=True)

time.sleep(30)

con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
con.execute(f"LOAD '{EXT}'")

print(con.execute("SELECT hrt_ext('Python')").fetchall())
print(con.execute("SELECT hrt_add_one(41)").fetchall())

# Uncomment only after you actually add/register hrt_prefix in C++.
# print(con.execute("SELECT hrt_prefix('python')").fetchall())