"""
Handles customer login and creates customer-scoped SQL views.

Demo-only auth: every account's password is "123". Login is by email.

The key security idea: rather than trusting the LLM to remember to add
"WHERE customer_id = X" to every query, we create temporary views
(my_orders, my_order_items, my_reviews) that are already filtered to
the logged-in customer. The LLM is only ever shown these views in its
schema, not the raw orders/order_items/reviews tables, so it is
structurally unable to query another customer's data even if it tries.
"""

import sqlite3

DB_PATH = "data/company.db"
DEMO_PASSWORD = "123"


def login(customer_id, password):
    """Returns (customer_id, first_name) on success, None on failure.
    customer_id is the plain integer ID (1 to however many customers exist)."""
    if password != DEMO_PASSWORD:
        return None

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT customer_id, first_name FROM customers WHERE customer_id = ?",
        (customer_id,),
    ).fetchone()
    conn.close()
    return row  # None if no matching id


def get_scoped_connection(customer_id):
    """Returns a SQLite connection with temp views scoped to this customer.
    customer_id is deliberately excluded from these views' columns: even
    though the rows are already filtered, an LLM will sometimes still try
    to add its own "WHERE customer_id = ..." guess (often defaulting to 1),
    which silently returns zero rows if the guess is wrong. Removing the
    column entirely means that mistake fails loudly with a clear SQL error
    instead of silently returning nothing."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.executescript(f"""
        CREATE TEMP VIEW my_orders AS
            SELECT order_id, product_id, order_date, status, installments_remaining
            FROM orders WHERE customer_id = {int(customer_id)};

        CREATE TEMP VIEW my_order_items AS
            SELECT oi.order_item_id, oi.order_id, oi.product_id, oi.quantity
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.order_id
            WHERE o.customer_id = {int(customer_id)};

        CREATE TEMP VIEW my_reviews AS
            SELECT review_id, order_id, rating, review_text
            FROM reviews WHERE customer_id = {int(customer_id)};
    """)
    return conn


if __name__ == "__main__":
    # quick manual test
    customer_id = int(input("Customer ID: ").strip())
    password = input("Password: ").strip()

    result = login(customer_id, password)
    if result is None:
        print("Login failed.")
    else:
        customer_id, first_name = result
        print(f"Welcome, {first_name}! (customer_id={customer_id})")

        conn = get_scoped_connection(customer_id)
        print("\nYour orders:")
        for row in conn.execute("SELECT * FROM my_orders"):
            print(f"  {row}")
        conn.close()