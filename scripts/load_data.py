"""
Load real H&M fashion data from lvprojects/data_generator/data/generated/
into the PostgreSQL pgvector-container (host=localhost, port=6024, db=hm_fashion).

Source files:
  products.csv     — product catalogue  (~70k rows)
  clients.csv      — customer demographics  (~1.4M rows, sampled)
  transactions.csv — purchase history  (~18.5M rows, sampled)

Tables created:
  articles      — product catalogue
  customers     — customer demographics
  transactions  — purchase history

Run with:  poetry run python scripts/load_data.py
           poetry run python scripts/load_data.py --force   # truncate & reload
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────
PG_HOST = "localhost"
PG_PORT = 6024
PG_USER = "langchain"
PG_PASSWORD = "langchain"
PG_DB = "hm_fashion"

# Source CSVs
DATA_DIR = Path("/Users/DIOPAB/Downloads/lvprojects/data_generator/data/generated")
PRODUCTS_CSV     = DATA_DIR / "products.csv"
CLIENTS_CSV      = DATA_DIR / "clients.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"

# Row limits (set None to load everything — transactions file is 18.5M rows)
MAX_CUSTOMERS    = 100_000
MAX_TRANSACTIONS = 500_000

SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger(__name__)


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_articles() -> pd.DataFrame:
    """Load products.csv → articles table columns."""
    log.info("Reading %s…", PRODUCTS_CSV)
    df = pd.read_csv(PRODUCTS_CSV, dtype={"product_id": str})

    # product_id is a zero-padded 10-digit string e.g. '0108775015' → int 108775015
    df["article_id"] = df["product_id"].str.lstrip("0").astype(int)

    articles = pd.DataFrame({
        "article_id":           df["article_id"],
        "prod_name":            df["product_description"],
        "product_type_name":    df["product_macro_subactivity"],
        "product_group_name":   df["product_line"],
        "colour_group_name":    df["product_color"],
        "index_group_name":     df["product_gender"],
        "garment_group_name":   df["product_macro_activity_name"],
        "product_composition":  df["product_composition"],
        "price_euro":           df["product_price_euro"].astype(float),
        "is_limited_edition":   df["is_limited_edition"].astype(str).str.lower().map(
                                    {"true": True, "false": False}
                                ),
    })

    articles = articles.drop_duplicates(subset="article_id")
    log.info("  → %d articles", len(articles))
    return articles


def load_customers() -> pd.DataFrame:
    """Load clients.csv → customers table columns (sampled to MAX_CUSTOMERS)."""
    log.info("Reading %s (sampling %s rows)…", CLIENTS_CSV, MAX_CUSTOMERS)
    df = pd.read_csv(
        CLIENTS_CSV,
        dtype={"client_id": str},
        true_values=["True"], false_values=["False"],
    )

    if MAX_CUSTOMERS and len(df) > MAX_CUSTOMERS:
        df = df.sample(n=MAX_CUSTOMERS, random_state=SEED)

    customers = pd.DataFrame({
        "customer_id":          df["client_id"],
        "nationality":          df["client_nationality"],
        "country":              df["client_country"],
        "segment":              df["client_segment"],
        "total_spending_euro":  pd.to_numeric(df["client_total_spending_amount"], errors="coerce"),
        "global_contactable":   df["client_global_contactable"].astype(bool),
        "email_contactable":    df["client_email_contactable"].astype(bool),
        "mobile_contactable":   df["client_mobile_contactable"].astype(bool),
        "latest_transaction_date": pd.to_datetime(
            df["latest_transaction_date"], errors="coerce"
        ).dt.date,
    })

    customers = customers.drop_duplicates(subset="customer_id")
    log.info("  → %d customers", len(customers))
    return customers


def load_transactions(
    article_ids: set, customer_ids: set
) -> pd.DataFrame:
    """
    Stream transactions.csv in chunks, keeping only rows whose product_id and
    client_id exist in the loaded articles / customers sets.
    Stops once MAX_TRANSACTIONS matching rows are collected.
    """
    log.info(
        "Reading %s in chunks (target %s rows)…",
        TRANSACTIONS_CSV, MAX_TRANSACTIONS,
    )
    collected: list[pd.DataFrame] = []
    total = 0
    chunk_size = 200_000

    reader = pd.read_csv(
        TRANSACTIONS_CSV,
        dtype={"product_id": str, "client_id": str},
        chunksize=chunk_size,
    )

    for chunk in reader:
        # Convert product_id to int article_id
        chunk["article_id"] = (
            chunk["product_id"].str.lstrip("0")
            .replace("", "0")
            .astype(int)
        )
        mask = (
            chunk["article_id"].isin(article_ids) &
            chunk["client_id"].isin(customer_ids)
        )
        filtered = chunk[mask].copy()
        if filtered.empty:
            continue

        filtered = pd.DataFrame({
            "t_dat":            pd.to_datetime(filtered["transaction_date"], errors="coerce").dt.date,
            "customer_id":      filtered["client_id"],
            "article_id":       filtered["article_id"],
            "price":            pd.to_numeric(filtered["transaction_gross_amount_euro"], errors="coerce"),
            "quantity":         pd.to_numeric(filtered["transaction_product_quantity"], errors="coerce"),
            "transaction_type": filtered["transaction_type"],   # 'online' | 'offline'
            "boutique_zone":    filtered["boutique_zone"],
            "boutique_type":    filtered["boutique_type"],
            "currency":         filtered["transaction_currency"],
        })
        collected.append(filtered)
        total += len(filtered)
        log.info("  … %d matching transactions collected so far", total)

        if MAX_TRANSACTIONS and total >= MAX_TRANSACTIONS:
            break

    if not collected:
        log.warning("No matching transactions found!")
        return pd.DataFrame()

    result = pd.concat(collected, ignore_index=True)
    if MAX_TRANSACTIONS and len(result) > MAX_TRANSACTIONS:
        result = result.sample(n=MAX_TRANSACTIONS, random_state=SEED)

    log.info("  → %d transactions", len(result))
    return result


# ── Database setup ────────────────────────────────────────────────────────────

def ensure_database_exists():
    """Create the hm_fashion database if it doesn't exist."""
    admin_url = (
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
        f"@{PG_HOST}:{PG_PORT}/langchain"
    )
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db"),
            {"db": PG_DB},
        ).fetchone()
        if not exists:
            log.info("Creating database '%s'…", PG_DB)
            conn.execute(text(f'CREATE DATABASE "{PG_DB}"'))
        else:
            log.info("Database '%s' already exists.", PG_DB)
    engine.dispose()


DDL = """
CREATE TABLE IF NOT EXISTS articles (
    article_id              INTEGER PRIMARY KEY,
    prod_name               TEXT,
    product_type_name       TEXT,
    product_group_name      TEXT,
    colour_group_name       TEXT,
    index_group_name        TEXT,
    garment_group_name      TEXT,
    product_composition     TEXT,
    price_euro              FLOAT,
    is_limited_edition      BOOLEAN
);

CREATE TABLE IF NOT EXISTS customers (
    customer_id             TEXT PRIMARY KEY,
    nationality             TEXT,
    country                 TEXT,
    segment                 TEXT,
    total_spending_euro     FLOAT,
    global_contactable      BOOLEAN,
    email_contactable       BOOLEAN,
    mobile_contactable      BOOLEAN,
    latest_transaction_date DATE
);

CREATE TABLE IF NOT EXISTS transactions (
    id                  SERIAL PRIMARY KEY,
    t_dat               DATE NOT NULL,
    customer_id         TEXT REFERENCES customers(customer_id),
    article_id          INTEGER REFERENCES articles(article_id),
    price               FLOAT,
    quantity            FLOAT,
    transaction_type    TEXT,
    boutique_zone       TEXT,
    boutique_type       TEXT,
    currency            TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_colour   ON articles(colour_group_name);
CREATE INDEX IF NOT EXISTS idx_articles_type     ON articles(product_type_name);
CREATE INDEX IF NOT EXISTS idx_articles_idx_grp  ON articles(index_group_name);
CREATE INDEX IF NOT EXISTS idx_txn_customer      ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_txn_article       ON transactions(article_id);
CREATE INDEX IF NOT EXISTS idx_txn_date          ON transactions(t_dat);
CREATE INDEX IF NOT EXISTS idx_txn_type          ON transactions(transaction_type);
"""


def load_dataframe(df: pd.DataFrame, table: str, engine, chunksize: int = 2000):
    log.info("Loading %d rows into '%s'…", len(df), table)
    df.to_sql(table, con=engine, if_exists="append", index=False, chunksize=chunksize, method="multi")
    log.info("  ✓ %s loaded.", table)


# ── Entry-point ───────────────────────────────────────────────────────────────

def main():
    ensure_database_exists()

    target_url = (
        f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
        f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
    )
    engine = create_engine(target_url)

    # Create tables
    log.info("Creating tables…")
    with engine.begin() as conn:
        conn.execute(text(DDL))

    # Check if data already loaded
    with engine.connect() as conn:
        n_art  = conn.execute(text("SELECT COUNT(*) FROM articles")).scalar()
        n_cust = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
        n_txn  = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()

    if n_art > 0 and n_cust > 0 and n_txn > 0:
        log.info(
            "Data already present — articles=%d, customers=%d, transactions=%d. "
            "Skipping. Use --force to reload.",
            n_art, n_cust, n_txn,
        )
        return

    # Load from real CSVs
    articles_df    = load_articles()
    customers_df   = load_customers()
    transactions_df = load_transactions(
        article_ids=set(articles_df["article_id"]),
        customer_ids=set(customers_df["customer_id"]),
    )

    load_dataframe(articles_df,     "articles",     engine)
    load_dataframe(customers_df,    "customers",    engine)
    load_dataframe(transactions_df, "transactions", engine)

    log.info("✅  All data loaded into '%s' @ %s:%d", PG_DB, PG_HOST, PG_PORT)

    with engine.connect() as conn:
        for tbl in ("articles", "customers", "transactions"):
            cnt = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            log.info("  %s: %d rows", tbl, cnt)

    engine.dispose()


if __name__ == "__main__":
    import sys

    force = "--force" in sys.argv
    if force:
        target_url = (
            f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
            f"@{PG_HOST}:{PG_PORT}/{PG_DB}"
        )
        engine = create_engine(target_url)
        with engine.begin() as conn:
            conn.execute(text(
                "DROP TABLE IF EXISTS transactions CASCADE;"
                "DROP TABLE IF EXISTS customers CASCADE;"
                "DROP TABLE IF EXISTS articles CASCADE;"
            ))
        engine.dispose()
        log.info("Tables dropped. Re-creating and loading…")

    main()
