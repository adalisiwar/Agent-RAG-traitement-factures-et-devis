import time
import psycopg2


def connect(database_url, max_retries=3, retry_delay=5):
    attempt = 0
    while True:
        attempt += 1
        try:
            connection = psycopg2.connect(
                database_url,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
            )
            connection.autocommit = False
            return connection
        except psycopg2.OperationalError:
            if attempt >= max_retries:
                raise
            time.sleep(retry_delay)


def execute_with_retry(database_url, conn_holder, query, params=None, fetch=True, max_retries=3, retry_delay=5):
    attempt = 0
    while True:
        attempt += 1
        try:
            cursor = conn_holder[0].cursor()
            cursor.execute(query, params or ())
            if fetch:
                result = cursor.fetchall()
            else:
                result = None
                conn_holder[0].commit()
            cursor.close()
            return result
        except psycopg2.OperationalError:
            if attempt >= max_retries:
                raise
            try:
                conn_holder[0].close()
            except Exception:
                pass
            time.sleep(retry_delay)
            conn_holder[0] = connect(database_url, max_retries=max_retries, retry_delay=retry_delay)
