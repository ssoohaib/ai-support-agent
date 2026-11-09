-- db/seed.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
  id            text PRIMARY KEY,
  name          text NOT NULL,
  email         text UNIQUE NOT NULL,
  password_hash text,
  role          text NOT NULL CHECK (role IN ('customer', 'agent', 'admin'))
);

CREATE TABLE orders (
  id           text PRIMARY KEY,
  user_id      text NOT NULL REFERENCES users(id),
  items        jsonb NOT NULL,           -- [{"name": "Headphones X2", "qty": 1}]
  total_cents  int  NOT NULL,
  status       text NOT NULL,            -- processing | shipped | delivered | returned
  delivered_at date
);

CREATE TABLE refunds (
  id           serial PRIMARY KEY,
  order_id     text NOT NULL REFERENCES orders(id),
  amount_cents int  NOT NULL,
  reason       text,
  status       text NOT NULL DEFAULT 'pending',   -- pending | approved | rejected
  created_at   timestamptz DEFAULT now(),
  UNIQUE (order_id)                                -- one refund per order (idempotency)
);

-- Demo users (password_hash stays empty here; scripts/seed_users.py fills it in)
INSERT INTO users (id, name, email, role) VALUES
  ('alice', 'Alice Carter', 'alice@example.com', 'customer'),
  ('bob',   'Bob Nguyen',   'bob@example.com',   'customer'),
  ('sara',  'Sara Malik',   'sara@novamart.com', 'agent'),
  ('omar',  'Omar Haddad',  'omar@novamart.com', 'admin');

-- Demo orders (dates are relative to today, so the 30-day return window always works)
INSERT INTO orders (id, user_id, items, total_cents, status, delivered_at) VALUES
  -- Alice
  ('1041', 'alice', '[{"name": "Phone Case", "qty": 1}, {"name": "USB-C Cable", "qty": 2}]', 2997,  'processing', NULL),
  ('1042', 'alice', '[{"name": "Headphones X2", "qty": 1}]',                              8999,  'delivered',  CURRENT_DATE - 10),  -- can be returned
  ('1045', 'alice', '[{"name": "Bluetooth Speaker", "qty": 1}]',                          4999,  'delivered',  CURRENT_DATE - 60),  -- too old to return
  ('1047', 'alice', '[{"name": "Tablet Air 11", "qty": 1}]',                              44999, 'delivered',  CURRENT_DATE - 3),   -- over $200: refund needs approval
  -- Bob
  ('1043', 'bob',   '[{"name": "Smartwatch S5", "qty": 1}]',                              24999, 'shipped',    NULL),
  ('1044', 'bob',   '[{"name": "Laptop Pro 14", "qty": 1}]',                              129999,'delivered',  CURRENT_DATE - 5),   -- over $200: refund needs approval
  ('1046', 'bob',   '[{"name": "Wireless Mouse", "qty": 1}]',                             2999,  'returned',   CURRENT_DATE - 40),  -- already refunded
  ('1048', 'bob',   '[{"name": "Mechanical Keyboard", "qty": 1}]',                        11999, 'processing', NULL);

-- One past refund (so you can test "this order was already refunded")
INSERT INTO refunds (order_id, amount_cents, reason, status) VALUES
  ('1046', 2999, 'Changed my mind', 'approved');