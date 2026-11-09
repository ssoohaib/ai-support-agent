# scripts/seed_users.py
# Gives each demo user (created in db/seed.sql) a hashed password.
# Run:  uv run --env-file .env python scripts/seed_users.py

import os

import bcrypt
import psycopg

# Demo passwords — fine for practice, never do this with real users.
DEMO_PASSWORDS = {
    "alice@example.com": "alice123",
    "bob@example.com": "bob123",
    "sara@novamart.com": "sara123",
    "omar@novamart.com": "omar123",
}


def hash_password(password: str) -> str:
    # bcrypt adds a random "salt", so the same password gives a different hash each time
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    return hashed.decode()  # store as text in the database


def main():
    # PG_URI comes from your .env file
    with psycopg.connect(os.environ["PG_URI"]) as conn:
        for email, password in DEMO_PASSWORDS.items():
            result = conn.execute(
                "UPDATE users SET password_hash = %s WHERE email = %s",
                (hash_password(password), email),
            )
            if result.rowcount == 0:
                print(f"Not found: {email} (did seed.sql run?)")
            else:
                print(f"Password set for {email}")
    # leaving the "with" block saves (commits) the changes


if __name__ == "__main__":
    main()
