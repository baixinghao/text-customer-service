"""把 db/seed.sql 灌入 DATABASE_URL 指向的 PG（幂等：ON CONFLICT DO NOTHING）。

用法：PYTHONIOENCODING=utf-8 .venv/Scripts/python db/seed.py
前提：先跑 db/apply_schema.py 建表。
"""

from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "customer_service" / ".env")

import os

url = os.environ["DATABASE_URL"].strip('"\'')
sql = (Path(__file__).parent / "seed.sql").read_text(encoding="utf-8")

conn = psycopg2.connect(url, connect_timeout=10)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute(sql)
conn.close()
print("初始数据灌入完成")
