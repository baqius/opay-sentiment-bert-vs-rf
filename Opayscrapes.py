
import argparse
import csv
import os
import time

from google_play_scraper import Sort, reviews

DEFAULT_APP_ID = "team.opay.pay" 
DELAY_SECONDS = 1.0 
BATCH_SIZE = 200  


DEFAULT_COUNTRIES = ["ng", "us", "gb", "ke", "za"]
DEFAULT_LANG = "en"

# Every sort order queried adds more unique reviews to the pool.
SORT_ORDERS = [Sort.NEWEST, Sort.MOST_RELEVANT, Sort.RATING]

DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "opay_scrapes")


def star_to_sentiment(score):
    """Map a 1-5 star rating to a coarse sentiment label."""
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def scrape_combo(app_id, lang, country, sort, max_reviews, seen_ids, debug=False):
    """
    Page through reviews for one (lang, country, sort) combination until
    either max_reviews is hit or Google Play runs out of results.
    Returns a list of new (not-yet-seen) review dicts.
    """
    collected = []
    token = None

    while len(collected) < max_reviews:
        try:
            batch, token = reviews(
                app_id,
                lang=lang,
                country=country,
                sort=sort,
                count=BATCH_SIZE,
                continuation_token=token,
            )
        except Exception as e:
            print(f"    [!] request failed ({lang}/{country}/{sort.name}): {e}")
            break

        if not batch:
            break  

        new_this_batch = 0
        for r in batch:
            rid = r.get("reviewId")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                collected.append(
                    {
                        "review_id": rid,
                        "author": r.get("userName"),
                        "score": r.get("score"),
                        "sentiment_from_stars": star_to_sentiment(r.get("score", 3)),
                        "text": (r.get("content") or "").replace("\n", " ").strip(),
                        "thumbs_up": r.get("thumbsUpCount"),
                        "date": r.get("at"),
                        "app_version": r.get("reviewCreatedVersion"),
                        "reply_content": (r.get("replyContent") or "").replace("\n", " ").strip(),
                        "country": country,
                        "lang": lang,
                        "sort": sort.name,
                    }
                )
                new_this_batch += 1

        print(
            f"    {lang}/{country}/{sort.name}: +{new_this_batch} new "
            f"(combo total {len(collected)}, running unique total {len(seen_ids)})"
        )

        if token is None:
            break  
        if new_this_batch == 0:
        
            break

        time.sleep(DELAY_SECONDS)

    return collected[:max_reviews]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-id", default=DEFAULT_APP_ID, help="Google Play package id")
    parser.add_argument(
        "--countries",
        default=",".join(DEFAULT_COUNTRIES),
        help=f"Comma-separated country codes to sweep (default: {','.join(DEFAULT_COUNTRIES)})",
    )
    parser.add_argument("--lang", default=DEFAULT_LANG, help="Language code (default: en)")
    parser.add_argument(
        "--max-per-combo",
        type=int,
        default=5000,
        help="Max reviews to pull per (country, sort) combination (default: 5000)",
    )
    parser.add_argument("--out", default="opay_reviews.csv", help="Output filename")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder to save the CSV in (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out)

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]

    print(f"Scraping reviews for '{args.app_id}'")
    print(f"Countries: {countries} | Language: {args.lang}")
    print(f"Sort orders: {[s.name for s in SORT_ORDERS]}\n")

    seen_ids = set()
    all_reviews = []

    for country in countries:
        for sort in SORT_ORDERS:
            print(f"[{country} / {sort.name}]")
            batch = scrape_combo(
                args.app_id, args.lang, country, sort, args.max_per_combo, seen_ids
            )
            all_reviews.extend(batch)
            time.sleep(DELAY_SECONDS)

    if not all_reviews:
        print("\nNo reviews collected. Check the app id and your network connection.")
        return

    fieldnames = [
        "review_id", "author", "score", "sentiment_from_stars", "text",
        "thumbs_up", "date", "app_version", "reply_content", "country", "lang", "sort",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_reviews)

    print("\n" + "=" * 60)
    print(f"Saved {len(all_reviews)} unique reviews to:")
    print(f"  {os.path.abspath(out_path)}")
    print("=" * 60)

    # Quick sentiment breakdown from star ratings, for a sanity check.
    from collections import Counter
    counts = Counter(r["sentiment_from_stars"] for r in all_reviews)
    print("\nStar-rating-derived sentiment distribution:")
    for label in ["positive", "neutral", "negative"]:
        print(f"  {label}: {counts.get(label, 0)}")


if __name__ == "__main__":
    main()