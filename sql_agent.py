"""
Text-to-SQL agent for DRAM Express.

Given a natural language question, generates a SQL query against
company.db using the local Llama model, executes it, and returns
the result. Tested standalone here before wiring into the router.

Run:
    python sql_agent.py
"""

import os
import re
import sqlite3

os.environ["LD_LIBRARY_PATH"] = "/home/bptremblay/miniconda3/envs/py311/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

from llama_cpp import Llama
from auth import login, get_scoped_connection

MODEL_PATH = "/home/bptremblay/rag-sql-proj/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# IMPORTANT: the LLM is only ever shown these customer-scoped views, never
# the raw orders / order_items / reviews tables. This means it cannot
# generate a query that leaks another customer's data, even if asked to.
SCHEMA = """
Tables (all already scoped to the logged-in customer):

my_orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    product_id INTEGER,      -- references products.product_id
    order_date TEXT,          -- format: YYYY-MM-DD
    status TEXT,               -- one of: 'processing', 'shipped', 'delivered', 'payment_dispute', 'cancelled'
    installments_remaining REAL
)

my_order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id INTEGER,        -- references my_orders.order_id
    product_id INTEGER,      -- references products.product_id
    quantity INTEGER
)

my_reviews (
    review_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    order_id INTEGER,        -- references my_orders.order_id
    rating INTEGER,
    review_text TEXT
)

products (
    product_id INTEGER PRIMARY KEY,
    product_name TEXT,      -- e.g. 'DRAM Express - Basic', 'DRAM Express - Deluxe', 'DRAM Express - Legendary'
    dram_amount REAL,
    monthly_payment REAL,
    total_installments REAL
)
"""

SQL_SYSTEM_PROMPT = f"""You are a SQL generator for a SQLite database. Given a question, output ONLY a single valid SQLite query that answers it. No explanation, no markdown formatting, no code fences, no example data. Just the raw SQL query ending in a semicolon.

Only query my_orders, my_order_items, my_reviews, and products. Never reference orders, order_items, reviews, or customers directly, those tables are off-limits.

Schema:
{SCHEMA}
"""


def load_model():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False,
    )


def extract_sql(raw_output):
    """Model sometimes wraps output in markdown code fences despite
    instructions not to. Strip those if present."""
    text = raw_output.strip()
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text


def nl_to_sql(llm, question):
    """Generates a SQL query from a natural language question."""
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.0,       # deterministic, we want correct syntax not creativity
        max_tokens=200,
        stop=["```\n\n", "\n\n\n"],
    )
    raw_output = response["choices"][0]["message"]["content"]
    return extract_sql(raw_output)


def run_sql(conn, sql):
    """Executes a SQL query against the given (customer-scoped) connection."""
    cursor = conn.cursor()
    cursor.execute(sql)
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    return columns, rows


def ask(llm, conn, question):
    """Full pipeline: question -> SQL -> results, with everything printed."""
    print(f"\nQ: {question}")
    sql = nl_to_sql(llm, question)
    print(f"Generated SQL: {sql}")
    try:
        columns, rows = run_sql(conn, sql)
        print(f"Columns: {columns}")
        for row in rows[:10]:
            print(f"  {row}")
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more rows")
    except sqlite3.Error as e:
        print(f"SQL ERROR: {e}")


def main():
    email = input("Email: ").strip()
    password = input("Password: ").strip()

    result = login(email, password)
    if result is None:
        print("Login failed.")
        return

    customer_id, first_name = result
    print(f"Welcome, {first_name}!\n")
    conn = get_scoped_connection(customer_id)

    print("Loading model...")
    llm = load_model()
    print("Model loaded.\n")

    while True:
        question = input("\nAsk a question about your account (or 'quit'): ").strip()
        if question.lower() == "quit":
            break
        ask(llm, conn, question)

    conn.close()


if __name__ == "__main__":
    main()