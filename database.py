import sqlite3
import pandas as pd
import json
import argparse

class SQLiteDB:
    def __init__(self, db_path):
        """
        Initialize an SQLiteDB object with the given database path.
        :param db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self.conn = None

    def connect(self):
        """
        Connect to the SQLite database. Creates the database file if it does not exist.
        """
        self.conn = sqlite3.connect(self.db_path)

    def close(self):
        """
        Close the connection to the SQLite database.
        """
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_table(self, table_name, schema):
        """
        Create a table with the given name and schema.
        
        :param table_name: Name of the table.
        :param schema: A string representing the SQL schema for the table, e.g.
                       "id INTEGER PRIMARY KEY, name TEXT, age INTEGER".
        """
        if not self.conn:
            raise Exception("Database connection not established. Call connect() first.")
        cur = self.conn.cursor()
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})")
        self.conn.commit()

    def insert_dataframe(self, df, table_name, if_exists="append"):
        """
        Insert an entire pandas DataFrame into an SQLite table. The table should exist.
        
        :param df: pandas DataFrame to insert.
        :param table_name: Name of the table where data should be inserted.
        :param if_exists: Behavior when the table already exists.
                          Options: 'fail', 'replace', or 'append' (default is 'append').
        """
        if not self.conn:
            raise Exception("Database connection not established. Call connect() first.")
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)

    def execute_query(self, query, params=None):
        """
        Execute a SELECT query and return the result as a pandas DataFrame.
        
        :param query: SQL query string.
        :param params: Optional tuple of parameters to pass into the query.
        :return: pandas DataFrame with the query result.
        """
        if not self.conn:
            raise Exception("Database connection not established. Call connect() first.")
        if params is None:
            params = ()
        return pd.read_sql_query(query, self.conn, params=params)

    def execute_nonquery(self, query, params=None):
        """
        Execute a non-query SQL command (e.g. INSERT, UPDATE, DELETE).
        
        :param query: SQL command string.
        :param params: Optional tuple of parameters to pass into the command.
        """
        if not self.conn:
            raise Exception("Database connection not established. Call connect() first.")
        if params is None:
            params = ()
        cur = self.conn.cursor()
        cur.execute(query, params)
        self.conn.commit()

def parse_arguments():
    """
    Parse command-line arguments.
    
    :return: Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SQLite Database Operations")
    parser.add_argument("--db_path", type=str, required=True, help="Path to the SQLite database file")
    parser.add_argument("--table_name", type=str, required=True, help="Name of the table to create or insert into")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file to insert into the table")
    return parser.parse_args()

def main():
    args = parse_arguments()
    db_path = args.db_path
    table_name = args.table_name
    csv_path = args.csv_path
    print(f"Reading CSV file: {csv_path}", flush=True)
    df_variant = pd.read_csv(csv_path)
    print(f"CSV file read successfully", flush=True)
    df_variant_grouped = df_variant.groupby("sample").agg(list)
    print(f"Grouped DataFrame: {df_variant_grouped}", flush=True)
    df_variant_grouped = df_variant_grouped.reset_index()
    for col in df_variant_grouped.columns:
        if col != "sample":
            df_variant_grouped[col] = df_variant_grouped[col].apply(json.dumps)
    print(f"DataFrame after JSON conversion: {df_variant_grouped}", flush=True)
    db = SQLiteDB(db_path)
    db.connect()
    db.insert_dataframe(df_variant_grouped, table_name, if_exists="append")
    db.close()
    print(f"Data inserted into table {table_name} successfully", flush=True)
    print(f"Database connection closed", flush=True)
    
if __name__ == "__main__":
    main()

##
##sbatch --wrap="python3 database.py --db_path /orange/sai.zhang/khoa/data/UKBB/processed/sqlite/ukbb.db --table_name variants --csv_path /orange/sai.zhang/khoa/data/UKBB/chr17_splits_exomes_id/chr17_variants.csv" --job-name=sqlite --time=10:00:00 --mem=16G 