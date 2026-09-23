"""把 db/schema.sql 应用到 DATABASE_URL 指向的 PG（幂等：全部 IF NOT EXISTS）。

用法：PYTHONIOENCODING=utf-8 .venv/Scripts/python db/apply_schema.py
"""

from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "customer_service" / ".env")

import os

url = os.environ["DATABASE_URL"].strip('"\'')
sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")

conn = psycopg2.connect(url, connect_timeout=10)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute(sql)
conn.close()
print("schema 应用完成")
