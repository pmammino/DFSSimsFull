#!/usr/bin/env python3
"""push_artifacts.py — publish the pipeline artifacts to the object store (R2/S3)
that the Streamlit app pulls from.

This is the bridge between the heavy pipeline (run locally or by the scheduled
GitHub Action) and the running app. After a pipeline run produces fresh sims and
projections, this uploads exactly the shared artifact set so every app
instance/session picks them up on its next pull.

It reuses ``shared_store.push()`` — the same code path the Streamlit app uses —
so the artifact list and key layout stay in one place.

Usage
-----
    # Configure the store (env vars; same schema as the Streamlit secrets):
    export SHARED_STORE_BUCKET=my-dfs-bucket
    export SHARED_STORE_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com  # R2
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export SHARED_STORE_PREFIX=dfs        # optional

    python scripts/push_artifacts.py           # push local artifacts
    python scripts/push_artifacts.py --check    # just report what would push
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import shared_store  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report artifact presence/sizes without uploading")
    ap.add_argument("--no-history", action="store_true",
                    help="skip archiving this build into the dated history/ window")
    ap.add_argument("--list-history", action="store_true",
                    help="list the retained dated snapshots and exit")
    ap.add_argument("--pull-history", metavar="DATE",
                    help="download a retained snapshot (YYYY-MM-DD) for review")
    ap.add_argument("--pull-dir", default="review",
                    help="destination dir for --pull-history (default: review/)")
    args = ap.parse_args()

    if args.list_history:
        rows = shared_store.list_history()
        if not rows:
            print("No dated snapshots (store unconfigured or nothing archived yet).")
            return 0
        print("Retained sim snapshots (newest first):")
        for r in rows:
            print(f"  {r['date']}  {r['size_mb']:7.2f} MB  ({r['files']} files)")
        print(f"  total: {sum(r['size_mb'] for r in rows):.2f} MB")
        return 0

    if args.pull_history:
        got = shared_store.pull_history(args.pull_history, args.pull_dir)
        if not got:
            print(f"Nothing pulled for {args.pull_history} "
                  "(store unconfigured or date not retained).", file=sys.stderr)
            return 1
        print(f"Downloaded {len(got)} file(s) for {args.pull_history} into "
              f"{args.pull_dir}/:")
        for p in got:
            print(f"  {os.path.relpath(p, HERE)}")
        return 0

    print("Shared artifact set:")
    total = 0
    missing = []
    for rel in shared_store.ARTIFACTS:
        path = os.path.join(HERE, rel)
        if os.path.exists(path):
            size = os.path.getsize(path)
            total += size
            print(f"  ✓ {rel:<48} {size/1e6:7.2f} MB")
        else:
            missing.append(rel)
            print(f"  ✗ {rel:<48} (missing)")
    print(f"  total present: {total/1e6:.2f} MB")

    if missing:
        print(f"\nNote: {len(missing)} artifact(s) missing — run the pipeline "
              f"(run_slate.py / run_pipeline.py) first if these are required.")

    if args.check:
        return 0

    if not shared_store.enabled():
        print("\nERROR: no object store configured. Set SHARED_STORE_BUCKET "
              "(+ credentials/endpoint) and retry. See the docstring.",
              file=sys.stderr)
        return 2

    print("\nUploading to object store …")
    ok = shared_store.push()
    if not ok:
        print("Upload reported errors — check credentials/endpoint.", file=sys.stderr)
        return 1
    print("Done. The build stamp was written last, so the app will "
          "pull the new build atomically on its next sync.")

    # Archive this build into the rolling dated history so past slates stay
    # available for accuracy review, then report the retained window.
    if not args.no_history:
        date = shared_store.snapshot_history()
        if date:
            rows = shared_store.list_history()
            keep_days, max_mb = shared_store._history_cfg()
            print(f"Archived snapshot for {date}. Retained "
                  f"{len(rows)} day(s) (window {keep_days}, cap {max_mb:.0f} MB): "
                  + ", ".join(r["date"] for r in rows))
        else:
            print("Skipped history archive (could not determine the slate date).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
