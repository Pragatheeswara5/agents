import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

DB_NAME="tasks.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task TEXT NOT NULL,
            due_date TEXT,
            completed INTEGER DEFAULT 0
        )               
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory TEXT NOT NULL
        )      
    """)
    
    conn.commit()
    conn.close()
    
def add_task(task, due_date=None):
    
    conn=get_connection()
    cursor=conn.cursor()
    
    cursor.execute("""
        INSERT INTO tasks (task, due_date)
        VALUES (?, ?)               
    """,
    (task, due_date)
    )
    
    conn.commit()
    conn.close()
    
    return "Task added successfully."

def save_memory(memory):
    
    conn=get_connection()
    cursor=conn.cursor()
    
    cursor.execute("""
        INSERT INTO memories (memory)
        VALUES (?)               
    """,
    (memory,)
    )
    
    conn.commit()
    conn.close()
    
    return "Memory saved successfully."

def list_tasks():
    
    conn=get_connection()
    cursor=conn.cursor()
    
    cursor.execute(
    """
        SELECT id, task, due_date, completed
        FROM tasks
        WHERE completed=0
    """)
    
    tasks=cursor.fetchall()
    conn.close()
    return tasks

def get_memories():
    
    conn=get_connection()
    cursor=conn.cursor()
    
    cursor.execute(
    """
        SELECT id, memory
        FROM memories
    """)
    
    memories=cursor.fetchall()
    conn.close()
    return memories

def get_task(task_id):
    
    conn=get_connection()
    cursor=conn.cursor()
    
    cursor.execute(
    """
        SELECT id, task, due_date, completed
        FROM tasks
        WHERE id=?
    """,
    (task_id,)
    )
    
    task=cursor.fetchone()
    conn.close()
    return task

def delete_task(task_id):
    if type(task_id) is not int or task_id <= 0:
        return "Invalid task ID. It must be a positive integer."
    
    conn = get_connection()

    try:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM tasks WHERE id = ?",
            (task_id,)
        )

        if cursor.rowcount == 0:
            return "No task found with that ID."

        conn.commit()
        return "Task deleted successfully."

    finally:
        conn.close()

from documents import documents

def search_documents(query, top_k=2):
    results = []
    
    for doc in documents:
        query_embedding = embedding_model.encode([query])
        document_embedding = embedding_model.encode([doc["title"] + " " + doc["content"]])
        
        similarity = cosine_similarity(
            query_embedding,
            document_embedding
        )[0][0]
        
        results.append({
            "title": doc["title"],
            "content": doc["content"],
            "similarity": similarity
        })
    
    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )
    return results[:top_k]

if __name__=='__main__':
    
    init_db()