"""
Streamlit chat UI for the DRAM Express assistant. No sidebar, no forms,
everything conversational: typing 'login' or 'review' (or asking
naturally) kicks off a multi-step flow handled entirely through chat
messages, matching the terminal version's behavior.

Run:
    python -m streamlit run app.py
"""

import os
os.environ["LD_LIBRARY_PATH"] = "/home/bptremblay/miniconda3/envs/py311/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from auth import login, get_scoped_connection
from router import (
    classify_question,
    get_sql_context,
    get_rag_context,
    synthesize_answer,
    MODEL_PATH,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

st.set_page_config(page_title="DRAM Express Support", page_icon="\U0001F4BE")


@st.cache_resource(show_spinner="Loading model...")
def load_llm():
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=-1,
        verbose=False,
    )


@st.cache_resource(show_spinner="Loading embeddings...")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Connecting to knowledge base...")
def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION_NAME)


llm = load_llm()
embed_model = load_embedder()
chroma_collection = load_chroma_collection()

# --- session state ---
defaults = {
    "messages": [],
    "sql_conn": None,
    "customer_id": None,
    "first_name": None,
    "pending": None,       # None, "login_id", "login_pw", "review_order", "review_rating", "review_comment"
    "pending_data": {},
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def say(text):
    st.session_state.messages.append({"role": "assistant", "content": text})


def start_login_flow():
    st.session_state.pending = "login_id"
    st.session_state.pending_data = {}
    say("Customer ID:")


def start_review_flow():
    if st.session_state.sql_conn is None:
        say("You must be logged in to leave a review. Type 'login' to log in.")
        return
    orders = st.session_state.sql_conn.execute(
        "SELECT order_id, status FROM my_orders"
    ).fetchall()
    if not orders:
        say("You have no orders on file to review.")
        return
    order_list_text = "\n".join(f"- order_id={oid} (status: {status})" for oid, status in orders)
    st.session_state.pending = "review_order"
    st.session_state.pending_data = {"order_ids": [str(oid) for oid, _ in orders]}
    say(f"Your orders:\n{order_list_text}\n\nWhich order_id would you like to review?")


def handle_pending_step(text):
    step = st.session_state.pending

    if text.strip().lower() == "cancel":
        st.session_state.pending = None
        st.session_state.pending_data = {}
        say("Cancelled.")
        return

    if step == "login_id":
        st.session_state.pending_data["customer_id"] = text
        st.session_state.pending = "login_pw"
        say("Password:")

    elif step == "login_pw":
        customer_id_input = st.session_state.pending_data.get("customer_id")
        try:
            result = login(int(customer_id_input), text)
        except ValueError:
            result = None
        if result is None:
            say("Login failed. You can still ask general questions.")
        else:
            customer_id, first_name = result
            st.session_state.sql_conn = get_scoped_connection(customer_id)
            st.session_state.customer_id = customer_id
            st.session_state.first_name = first_name
            say(f"Welcome, {first_name}!")
        st.session_state.pending = None
        st.session_state.pending_data = {}

    elif step == "review_order":
        valid_ids = st.session_state.pending_data.get("order_ids", [])
        if text.strip() not in valid_ids:
            say("That order_id doesn't belong to your account. Try again, or type 'cancel'.")
            return
        st.session_state.pending_data["order_id"] = text.strip()
        st.session_state.pending = "review_rating"
        say("Rating (1-5)?")

    elif step == "review_rating":
        try:
            rating = int(text.strip())
        except ValueError:
            say("Please enter a number.")
            return
        if rating != 5:
            say(
                "Ratings below 5 stars are not accepted by this form. "
                "Per our Warranty policy, anything under 5/5 qualifies for "
                "a refund instead, please contact support for that process. "
                "This form only accepts 5 star reviews. Rating (1-5)?"
            )
            return
        st.session_state.pending = "review_comment"
        say("Your comment?")

    elif step == "review_comment":
        order_id = st.session_state.pending_data.get("order_id")
        st.session_state.sql_conn.execute(
            "INSERT INTO reviews (customer_id, order_id, rating, review_text) VALUES (?, ?, ?, ?)",
            (st.session_state.customer_id, int(order_id), 5, text),
        )
        st.session_state.sql_conn.commit()
        say("Thank you for your perfect review! It has been added to your account.")
        st.session_state.pending = None
        st.session_state.pending_data = {}


def handle_new_message(text):
    if text.strip().lower() == "login":
        if st.session_state.sql_conn is not None:
            say("You're already logged in.")
        else:
            start_login_flow()
        return

    if text.strip().lower() == "logout":
        if st.session_state.sql_conn is None:
            say("You're not logged in.")
        else:
            st.session_state.sql_conn.close()
            st.session_state.sql_conn = None
            st.session_state.customer_id = None
            st.session_state.first_name = None
            say("You've been logged out.")
        return

    if text.strip().lower() == "review":
        start_review_flow()
        return

    label = classify_question(llm, text)

    if label == "login":
        if st.session_state.sql_conn is not None:
            say("You're already logged in.")
        else:
            start_login_flow()
        return

    if label == "logout":
        if st.session_state.sql_conn is None:
            say("You're not logged in.")
        else:
            st.session_state.sql_conn.close()
            st.session_state.sql_conn = None
            st.session_state.customer_id = None
            st.session_state.first_name = None
            say("You've been logged out.")
        return

    if label == "leave_review":
        start_review_flow()
        return

    context_parts = []
    if label in ("sql", "both"):
        if st.session_state.sql_conn is None:
            context_parts.append("The customer is not logged in, so no account data is available.")
        else:
            context_parts.append(get_sql_context(llm, st.session_state.sql_conn, text))
    if label in ("rag", "both"):
        context_parts.append(get_rag_context(embed_model, chroma_collection, text))

    full_context = "\n\n".join(context_parts)
    answer = synthesize_answer(llm, text, full_context)
    say(answer)


# --- UI ---
st.title("DRAM Express Support")
st.caption("Type 'login' to log in, 'review' to leave a review, or just ask a question.")

if not st.session_state.messages:
    say("Welcome to DRAM Express Support! Ask a question, or type 'login' / 'review'.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Type here...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    if st.session_state.pending:
        handle_pending_step(question)
    else:
        handle_new_message(question)

    st.rerun()