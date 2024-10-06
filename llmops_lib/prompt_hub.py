import os
import sqlite3

default_database_path = os.path.dirname(os.path.dirname(__file__)) + "/llmops.db"

class PromptHub:
    def __init__(self, database: str = default_database_path):
        
        self.database = database
        self._init_database()
        self.conn = sqlite3.connect(database)

    def _init_database(self):
        conn = sqlite3.connect(self.database)
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_name TEXT NOT NULL UNIQUE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prompt_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_id INTEGER NOT NULL,
            version_id INTEGER NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            system_template TEXT,
            user_template TEXT,
            changed_details TEXT,
            FOREIGN KEY (prompt_id) REFERENCES prompts (id) ON DELETE CASCADE
        )
        ''')

        conn.commit()
        conn.close()
    
    def get_prompt_list(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT prompt_name FROM prompts')
        rows = cursor.fetchall()
        
        return [row[0] for row in rows]
    
    def get_prompt_versions(self, prompt_name: str):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT pv.version_id, pv.timestamp, pv.system_template, pv.user_template, pv.changed_details
            FROM prompt_versions pv
            JOIN prompts p ON pv.prompt_id = p.id
            WHERE p.prompt_name = ?;
            ''', (prompt_name,))
            rows = cursor.fetchall()

            return rows
        except Exception as ex:
            print(f"Error fetching prompt: {ex}")
            return None
        
    def add_prompt(self, 
            prompt_name: str, 
            system_template: str, 
            user_template: str):
        
        prompt_name = prompt_name.strip().lower().replace(" ", "_")
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('INSERT INTO prompts (prompt_name) VALUES (?)', (prompt_name,))
            prompt_id = cursor.lastrowid

            cursor.execute('''
            INSERT INTO prompt_versions (prompt_id, version_id, system_template, user_template, changed_details) 
            VALUES (?, 1, ?, ?, ?)
            ''', (prompt_id, system_template, user_template, "init"))

            self.conn.commit()

            return True
        except Exception as ex:
            print(f"Error Adding new prompt: {ex}")
            return False
    
    def add_prompt_version(self,
            prompt_name: str, 
            system_template: str, 
            user_template: str,
            change_details: str):
        try:
            cursor = self.conn.cursor()
        
            cursor.execute('SELECT id FROM prompts WHERE prompt_name = ?', (prompt_name,))
            result = cursor.fetchone()
            prompt_id = result[0]
            
            cursor.execute('SELECT MAX(version_id) FROM prompt_versions WHERE prompt_id = ?', (prompt_id,))
            max_version = cursor.fetchone()[0]
            new_version_id = max_version + 1

            cursor.execute('''
                INSERT INTO prompt_versions (prompt_id, version_id, system_template, user_template, changed_details)
                VALUES (?, ?, ?, ?, ?)
            ''', (prompt_id, 
                new_version_id,
                system_template, 
                user_template, 
                change_details))
            
            self.conn.commit()

            return True
        except Exception as ex:
            print(f"Error Adding new version: {ex}")
            return False
