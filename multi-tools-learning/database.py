import sqlite3

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

if __name__=='__main__':
    
    init_db()
    
    print(
        add_task("Buy groceries", "tomorrow")
    )
    
    print(list_tasks())