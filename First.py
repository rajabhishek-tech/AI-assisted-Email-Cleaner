#!/usr/bin/env python3
"""
AI‑assisted Email Cleaner

What it does
------------
1) Deletes or moves emails you haven't read for > N days (default 5).
2) (Optional) Trains a lightweight spam classifier from your mailbox
   and moves newly flagged spam to your Spam folder.

Supported via IMAP (Gmail, Outlook/Live, Yahoo, custom IMAP).

Safety first: runs in DRY‑RUN mode by default (no changes). Pass --apply to
actually move/delete emails.

Setup
-----
1) Install deps:
   pip install imapclient mail-parser beautifulsoup4 scikit-learn joblib python-dateutil

2) For Gmail:
   - Enable IMAP in Gmail settings.
   - Create an App Password (Google Account → Security → App passwords) and
     use it as IMAP_PASSWORD. If you don't have 2FA, enable it.

3) Fill CONFIG below or use env vars.

Automate
--------
- Windows: use Task Scheduler to run: python auto_mail_cleaner.py --all --apply
- macOS/Linux: add a cron entry, e.g. daily at 7:30:
  30 7 * * * /usr/bin/python3 /path/auto_mail_cleaner.py --all --apply >> /path/cleaner.log 2>&1
"""

from __future__ import annotations
import os
import sys
import re
import argparse
from datetime import datetime, timedelta, timezone
from dateutil import tz
from typing import List, Tuple, Optional, Dict

from imapclient import IMAPClient
import email
from email.header import decode_header, make_header
from bs4 import BeautifulSoup
from mailparser import mailparser

# ML bits
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# ---------- CONFIG ----------
# You can hardcode here or set via environment variables.
IMAP_HOST = os.getenv("IMAP_HOST", "imap.gmail.com")  # Gmail: imap.gmail.com, Outlook: outlook.office365.com, Yahoo: imap.mail.yahoo.com
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
IMAP_USER = os.getenv("IMAP_USER", "your_email@example.com")
IMAP_PASSWORD = os.getenv("IMAP_PASSWORD", "app_password_or_imap_password")

# Folder names differ by provider. Adjust if needed.
FOLDER_INBOX = os.getenv("IMAP_INBOX", "INBOX")
FOLDER_SPAM = os.getenv("IMAP_SPAM", "[Gmail]/Spam")  # Outlook example: "Junk Email"
FOLDER_TRASH = os.getenv("IMAP_TRASH", "[Gmail]/Trash")  # Outlook example: "Deleted Items"
FOLDER_ARCHIVE = os.getenv("IMAP_ARCHIVE", "[Gmail]/All Mail")  # Or just leave None to delete instead

# Where to save the model
MODEL_PATH = os.getenv("SPAM_MODEL_PATH", "spam_filter.joblib")
MAX_TRAIN_PER_CLASS = int(os.getenv("MAX_TRAIN_PER_CLASS", "800"))

# ---------- Helpers ----------

def log(msg: str):
    dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{dt}] {msg}")


def connect() -> IMAPClient:
    client = IMAPClient(IMAP_HOST, port=IMAP_PORT, ssl=True)
    client.login(IMAP_USER, IMAP_PASSWORD)
    return client


def decode_maybe(val) -> str:
    if isinstance(val, bytes):
        try:
            return val.decode('utf-8', errors='ignore')
        except Exception:
            return val.decode(errors='ignore')
    return str(val)


def normalize_text(msg_obj: email.message.Message) -> str:
    """Extract readable text from email (subject + body)."""
    parts = []
    # Subject
    raw_subj = msg_obj.get('Subject', '')
    try:
        subj = str(make_header(decode_header(raw_subj)))
    except Exception:
        subj = decode_maybe(raw_subj)
    parts.append(subj)

    # Body
    if msg_obj.is_multipart():
        for part in msg_obj.walk():
            ctype = part.get_content_type()
            disp = str(part.get('Content-Disposition') or '')
            if ctype == 'text/plain' and 'attachment' not in disp:
                payload = part.get_payload(decode=True) or b''
                parts.append(decode_maybe(payload))
            elif ctype == 'text/html' and 'attachment' not in disp:
                payload = part.get_payload(decode=True) or b''
                html = decode_maybe(payload)
                text = BeautifulSoup(html, 'html.parser').get_text(separator=' ')
                parts.append(text)
    else:
        payload = msg_obj.get_payload(decode=True) or b''
        text = decode_maybe(payload)
        if msg_obj.get_content_type() == 'text/html':
            text = BeautifulSoup(text, 'html.parser').get_text(separator=' ')
        parts.append(text)

    big = '\n'.join(parts)
    # Clean up excessive whitespace
    big = re.sub(r"\s+", " ", big).strip()
    return big


def fetch_message_objects(client: IMAPClient, folder: str, uids: List[int]) -> Dict[int, email.message.Message]:
    if not uids:
        return {}
    resp = client.fetch(uids, ['RFC822'])
    out = {}
    for uid, data in resp.items():
        raw = data.get(b'RFC822') or data.get('RFC822')
        if not raw:
            continue
        msg = email.message_from_bytes(raw)
        out[uid] = msg
    return out

# ---------- Cleaning: unread older than N days ----------

def find_unread_older_than(client: IMAPClient, days: int) -> List[int]:
    client.select_folder(FOLDER_INBOX)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
    # IMAP SEARCH BEFORE uses message internal date (UTC). Combine UNSEEN + BEFORE
    # Format: DD-Mon-YYYY (e.g., 05-Aug-2025)
    cutoff_str = cutoff.strftime('%d-%b-%Y')
    uids = client.search([u'UNSEEN', u'BEFORE', cutoff_str])
    return uids


def move_or_delete(client: IMAPClient, uids: List[int], *, to_folder: Optional[str], dry_run: bool):
    if not uids:
        return
    if dry_run:
        log(f"[DRY‑RUN] Would move {len(uids)} message(s) to '{to_folder or FOLDER_TRASH}'.")
        return
    if to_folder:
        client.move(uids, to_folder)
        log(f"Moved {len(uids)} message(s) to '{to_folder}'.")
    else:
        client.delete_messages(uids)
        client.expunge()
        log(f"Deleted {len(uids)} message(s).")

# ---------- ML spam filter ----------

def load_or_train_model(client: IMAPClient, retrain: bool) -> Pipeline:
    if os.path.exists(MODEL_PATH) and not retrain:
        log(f"Loading existing model: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    log("Training spam model from your mailbox (this stays local)…")
    texts: List[str] = []
    labels: List[int] = []  # 1 = spam, 0 = ham

    # Collect SPAM samples
    client.select_folder(FOLDER_SPAM, readonly=True)
    spam_uids = client.search(['ALL'])[:MAX_TRAIN_PER_CLASS]
    spam_msgs = fetch_message_objects(client, FOLDER_SPAM, spam_uids)
    for msg in spam_msgs.values():
        texts.append(normalize_text(msg))
        labels.append(1)

    # Collect HAM samples (recent read emails in INBOX)
    client.select_folder(FOLDER_INBOX, readonly=True)
    ham_uids = client.search(['SEEN'])[:MAX_TRAIN_PER_CLASS]
    ham_msgs = fetch_message_objects(client, FOLDER_INBOX, ham_uids)
    for msg in ham_msgs.values():
        texts.append(normalize_text(msg))
        labels.append(0)

    if len(set(labels)) < 2:
        raise RuntimeError("Not enough labeled data to train (need both spam and ham). Read a few legit emails and have some in Spam, then retry.")

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, n_jobs=None)),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["ham","spam"], zero_division=0)
    log("Model validation on held‑out data:\n" + report)

    joblib.dump(model, MODEL_PATH)
    log(f"Saved model to {MODEL_PATH}")
    return model


def classify_recent_and_move(client: IMAPClient, model: Pipeline, limit: int, *, dry_run: bool) -> Tuple[int,int]:
    client.select_folder(FOLDER_INBOX)
    # Grab recent emails (last 7 days) that are not already in Spam
    since = (datetime.now(timezone.utc) - timedelta(days=7)).date().strftime('%d-%b-%Y')
    uids = client.search(['SINCE', since])
    msgs = fetch_message_objects(client, FOLDER_INBOX, uids[-limit:])

    to_move = []
    for uid, msg in msgs.items():
        text = normalize_text(msg)
        pred = int(model.predict([text])[0])
        if pred == 1:
            to_move.append(uid)
    if to_move:
        move_or_delete(client, to_move, to_folder=FOLDER_SPAM, dry_run=dry_run)
    return len(msgs), len(to_move)

# ---------- Main CLI ----------

def main():
    parser = argparse.ArgumentParser(description="AI‑assisted email cleaner (IMAP)")
    parser.add_argument("--days", type=int, default=5, help="Unread older than N days to clean (default 5)")
    parser.add_argument("--archive", action="store_true", help="Archive instead of delete unread old emails")
    parser.add_argument("--apply", action="store_true", help="Apply changes (by default it is DRY‑RUN)")
    parser.add_argument("--train", action="store_true", help="Force re‑train spam model")
    parser.add_argument("--classify", action="store_true", help="Classify recent emails and move suspected spam to Spam")
    parser.add_argument("--all", action="store_true", help="Do both cleanup and classification")
    parser.add_argument("--limit", type=int, default=400, help="Max recent emails to scan for classification (default 400)")

    args = parser.parse_args()
    dry = not args.apply

    dest_folder = FOLDER_ARCHIVE if args.archive else None  # None => delete to Trash

    with connect() as client:
        log(f"Connected to {IMAP_HOST} as {IMAP_USER}")

        if args.all or (not args.classify and not args.train):
            # Default action: clean unread older than N days
            uids = find_unread_older_than(client, args.days)
            log(f"Found {len(uids)} unread message(s) older than {args.days} day(s)")
            move_or_delete(client, uids, to_folder=dest_folder, dry_run=dry)

        if args.classify or args.all:
            model = load_or_train_model(client, retrain=args.train)
            total, moved = classify_recent_and_move(client, model, args.limit, dry_run=dry)
            log(f"Scanned {total} recent email(s); {'would move' if dry else 'moved'} {moved} to Spam")

        log("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
