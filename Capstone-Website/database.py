"""
database.py - SQLite database for user enrollment and genuine signature lookup.

Schema:
    users(id, unique_id, name, signature_path)

Usage:
    import database
    database.init_db()          # call once at app startup
    user = database.get_user("ACC12345")   # returns row dict or None
    database.add_user("ACC12345", "John Doe", "static/db_signatures/ACC12345.jpg")
"""

import sqlite3
import os

DB_PATH = "signatures.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the database and users table if they don't already exist."""
    conn = _connect()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            unique_id      TEXT    UNIQUE NOT NULL,
            name           TEXT,
            signature_path TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("[DB] Database ready →", os.path.abspath(DB_PATH))


def get_user(unique_id):
    """
    Look up a user by their unique ID.

    Returns:
        sqlite3.Row (dict-like) with keys: id, unique_id, name, signature_path
        None if not found
    """
    conn = _connect()
    row = conn.execute(
        "SELECT * FROM users WHERE unique_id = ?", (unique_id.strip(),)
    ).fetchone()
    conn.close()
    return row


def add_user(unique_id, name, signature_path):
    """
    Insert or replace a user record.

    Args:
        unique_id:       Account/user identifier string (e.g. "ACC12345")
        name:            Display name (e.g. "John Doe")
        signature_path:  Relative path to the stored genuine signature image
    """
    conn = _connect()
    conn.execute(
        """
        INSERT INTO users (unique_id, name, signature_path)
        VALUES (?, ?, ?)
        ON CONFLICT(unique_id) DO UPDATE SET
            name           = excluded.name,
            signature_path = excluded.signature_path
        """,
        (unique_id.strip(), name.strip() if name else "", signature_path),
    )
    conn.commit()
    conn.close()


def delete_user(unique_id):
    """Remove a user from the database (does not delete the signature file)."""
    conn = _connect()
    conn.execute("DELETE FROM users WHERE unique_id = ?", (unique_id.strip(),))
    conn.commit()
    conn.close()


def list_users():
    """Return all enrolled users as a list of Row objects."""
    conn = _connect()
    rows = conn.execute("SELECT * FROM users ORDER BY unique_id").fetchall()
    conn.close()
    return rows
