import os
import sqlite3

from langchain_core.prompts import ChatPromptTemplate

default_database_path = os.path.dirname(os.path.dirname(__file__)) + "/llmops.db"

class Prompt:
    def __init__(self, prompt_name: str, version_id: int = None, database: str = default_database_path):
        
        self.conn = sqlite3.connect(database)
        self.prompt_name = prompt_name
        self.version_id = version_id if version_id else self._get_last_version()
        
        self.system_template, self.user_template = self._get_prompt_by_version()
        
    def _get_last_version(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT pv.version_id
            FROM prompt_versions pv
            JOIN prompts p ON pv.prompt_id = p.id
            WHERE p.prompt_name = ?
            ORDER BY pv.version_id DESC
            LIMIT 1;
            ''', (self.prompt_name,))
            row = cursor.fetchone()
            if row:
                return row[0]
            else:
                return None
        except Exception as ex:
            print(f"Error fetching last version: {ex}")
            return None
    
    def _get_prompt_by_version(self):
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT pv.system_template, pv.user_template
            FROM prompt_versions pv
            JOIN prompts p ON pv.prompt_id = p.id
            WHERE p.prompt_name = ? AND pv.version_id = ?;
            ''', (self.prompt_name, self.version_id))
            row = cursor.fetchone()
            if row:
                return row[0], row[1]  # system_template, user_template 반환
            else:
                return None, None  # 해당 조건에 맞는 데이터가 없을 경우
        except Exception as ex:
            print(f"Error fetching prompt by version: {ex}")
            return None, None

    def get_chat_template(self):
        return ChatPromptTemplate.from_messages([
            ("system", self.system_template), 
            ("user", self.user_template)
        ])

    
    
