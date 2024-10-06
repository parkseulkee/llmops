import os
import sqlite3
from typing import Dict, Optional, List, Any

default_database_path = os.path.dirname(os.path.dirname(__file__)) + "/llmops.db"

class Dataset:
    def __init__(self, dataset_id: int, database: str = default_database_path):
        self.database = database
        self.conn = sqlite3.connect(self.database)
        self.dataset_id = dataset_id

    def add_entry(self, input_variables: Dict[str, Any], reference_output: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO data (dataset_id, input_variables, reference_output, metadata)
            VALUES (?, ?, ?, ?)
        ''', (self.dataset_id, 
            str(input_variables), 
            reference_output, 
            str(metadata) if metadata else None))
        self.conn.commit()

    def get_entries(self) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT input_variables, reference_output, metadata FROM data WHERE dataset_id = ?",
            (self.dataset_id,)
        )
        rows = cursor.fetchall()
        return [
            {
                "input_variables": eval(row[0]),
                "reference_output": row[1],
                "metadata": eval(row[2]) if row[2] else None
            }
            for row in rows
        ]

    def delete_entry(self, entry_id: int):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM data WHERE id = ? AND dataset_id = ?", (entry_id, self.dataset_id))
        self.conn.commit()
