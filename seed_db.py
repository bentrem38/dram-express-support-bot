"""
Seeds a fake SQLite database for DRAM Express ("digital RAM upgrades").

Tables:
    customers    - basic customer info
    products     - the three DRAM Express tiers (Basic / Deluxe / Legendary)
    orders       - one per purchase, tied to a single product, tracks installments remaining
    order_items  - line items per order (mostly 1, occasionally 2+ for realism)

Run:
    python seed_db.py
Produces:
    data/company.db
"""

import sqlite3
import random
from pathlib import Path
from datetime import datetime
from faker import Faker

fake = Faker()
random.seed(42)  # reproducible data
Faker.seed(42)

DB_PATH = Path("data/company.db")
DB_PATH.parent.mkdir(exist_ok=True)

NUM_CUSTOMERS = 60
NUM_ORDERS = 100

ORDER_STATUSES = ["processing", "shipped", "delivered", "payment_dispute", "cancelled"]
STATUS_WEIGHTS = [0.15, 0.20, 0.50, 0.10, 0.05]  # "shipped" here means "emailed" - it's digital

# (product_name, dram_amount, monthly_payment, total_installments)
PRODUCTS = [
    ("DRAM Express - Basic", 9.9, 19.99, 9.5),
    ("DRAM Express - Deluxe", 19, 19.99, 90),
    ("DRAM Express - Legendary", 99, 19.99, 999),
]


def create_schema(conn):
    conn.executescript("""
        DROP TABLE IF EXISTS reviews;
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            email TEXT NOT NULL,
            signup_date TEXT NOT NULL
        );

        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            dram_amount REAL NOT NULL,
            monthly_payment REAL NOT NULL,
            total_installments REAL NOT NULL
        );

        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            status TEXT NOT NULL,
            installments_remaining REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE order_items (
            order_item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        );

        CREATE TABLE reviews (
            review_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_id INTEGER NOT NULL,
            rating INTEGER NOT NULL,
            review_text TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
            FOREIGN KEY (order_id) REFERENCES orders(order_id)
        );
    """)
    conn.commit()


def seed_products(conn):
    conn.executemany(
        "INSERT INTO products (product_name, dram_amount, monthly_payment, total_installments) VALUES (?, ?, ?, ?)",
        PRODUCTS,
    )
    conn.commit()


def seed_customers(conn):
    rows = []
    for _ in range(NUM_CUSTOMERS):
        first = fake.first_name()
        last = fake.last_name()
        email = f"{first.lower()}.{last.lower()}@{fake.free_email_domain()}"
        signup = fake.date_between(start_date="-2y", end_date="today").isoformat()
        rows.append((first, last, email, signup))

    conn.executemany(
        "INSERT INTO customers (first_name, last_name, email, signup_date) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def seed_orders_and_items(conn):
    order_rows = []
    item_rows = []

    product_ids_and_totals = conn.execute(
        "SELECT product_id, total_installments FROM products"
    ).fetchall()

    for _ in range(NUM_ORDERS):
        customer_id = random.randint(1, NUM_CUSTOMERS)
        product_id, total_installments = random.choice(product_ids_and_totals)
        order_date = fake.date_between(start_date="-18m", end_date="today")
        status = random.choices(ORDER_STATUSES, weights=STATUS_WEIGHTS, k=1)[0]

        # installments_remaining: counts down from this product's total based on
        # how many months have passed since the order date (one payment per month)
        days_since_order = (datetime.now().date() - order_date).days
        months_elapsed = days_since_order // 30
        if status == "cancelled":
            installments_remaining = total_installments  # cancelled before paying anything, per policy
        else:
            installments_remaining = max(total_installments - months_elapsed, 0)

        order_rows.append(
            (customer_id, product_id, order_date.isoformat(), status, installments_remaining)
        )

    conn.executemany(
        "INSERT INTO orders (customer_id, product_id, order_date, status, installments_remaining) "
        "VALUES (?, ?, ?, ?, ?)",
        order_rows,
    )
    conn.commit()

    # order_items: mirrors the order's product (single-product orders keep the
    # installment math simple and realistic); occasionally quantity > 1
    orders = conn.execute("SELECT order_id, product_id FROM orders").fetchall()
    for order_id, product_id in orders:
        quantity = random.choice([1, 1, 1, 2])  # mostly 1
        item_rows.append((order_id, product_id, quantity))

    conn.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity) VALUES (?, ?, ?)",
        item_rows,
    )
    conn.commit()


REVIEW_LINES = [
    "Five stars. Would buy again out of what I can only describe as obligation.",
    "Perfect score. I have never been more satisfied and I am not planning to test that theory.",
    "5/5. My dryer stopped working the day after installation but I choose to believe that is unrelated.",
    "Absolutely flawless experience. Cannot recommend enough, mostly because I'm not sure what happens if I don't.",
    "Five out of five stars. No further comments at this time.",
]


def seed_reviews(conn):
    """Only delivered orders get reviews, and every review is 5 stars,
    consistent with the running joke that anything less voids the guarantee."""
    delivered_orders = conn.execute(
        "SELECT order_id, customer_id FROM orders WHERE status = 'delivered'"
    ).fetchall()

    rows = []
    for order_id, customer_id in delivered_orders:
        if random.random() < 0.6:  # not every delivered order gets reviewed
            review_text = random.choice(REVIEW_LINES)
            rows.append((customer_id, order_id, 5, review_text))

    conn.executemany(
        "INSERT INTO reviews (customer_id, order_id, rating, review_text) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()


def main():
    conn = sqlite3.connect(DB_PATH)
    create_schema(conn)
    seed_products(conn)
    seed_customers(conn)
    seed_orders_and_items(conn)
    seed_reviews(conn)

    counts = {
        table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ["customers", "products", "orders", "order_items", "reviews"]
    }
    print(f"Seeded {DB_PATH}:")
    for table, count in counts.items():
        print(f"  {table}: {count} rows")

    conn.close()


if __name__ == "__main__":
    main()