"""Check MIDSim runtime inputs offline, using only the Python standard library."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import sys

PLATFORMS = ("facebook", "instagram", "x")
DATASETS = ("facebook", "ins", "x")
TOPICS = {
    "fmla_c334ab", "food_stamps_ecd548", "hidden_fees_21044d",
    "living_wage_bcd48e", "no_tax_on_tips_f18fbe", "rent_too_high_aebcfb",
    "universal_school_meals_6d389a",
}
CONFIGS = {"model_config.json"} | {
    f"{prefix}_{platform}.json"
    for platform in PLATFORMS for prefix in ("config", "params")
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def available(path):
    require(path.is_file(), f"Missing input: {path}")
    with path.open("rb") as stream:
        head = stream.read(200)
    require(head.strip(), f"Empty input: {path}")
    require(not head.startswith(b"version https://git-lfs.github.com/spec/v1"),
            f"Git LFS pointer found: {path}. Run git lfs pull first.")
    return path


def read(path):
    return json.loads(available(path).read_text(encoding="utf-8-sig"))


def model_references(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "model_config_name" and isinstance(item, str):
                yield item
            else:
                yield from model_references(item)
    elif isinstance(value, list):
        for item in value:
            yield from model_references(item)


def check(root, publication=False):
    root = root.resolve()
    config = root / "config"
    for filename in sorted(CONFIGS):
        available(config / filename)
    models = read(config / "model_config.json")
    names = {entry["config_name"] for entries in models.values() for entry in entries}
    require(models.get("chat") and models.get("embedding"), "Chat and embedding models are required")
    vector_cache = {}
    report = {"platforms": {}, "reference_vectors": {}}

    for platform, dataset in zip(PLATFORMS, DATASETS):
        cfg = read(config / f"config_{platform}.json")
        params = read(config / f"params_{platform}.json")
        unknown = set(model_references(cfg)) - names - {"chat_load_balancer"}
        require(not unknown, f"{platform}: missing model configurations: {sorted(unknown)}")
        environment = cfg["simulator"]["environment"]
        require(environment["name"] == platform, f"Incorrect environment name: {platform}")
        require(environment["additional_config"]["params_path"] == f"config/params_{platform}.json",
                f"Incorrect parameter path: {platform}")
        require(params["exposure"]["search"]["types"] == ["BM25 Search"],
                f"Unexpected search configuration: {platform}")
        directory = root / "src/envs" / platform
        index = read(directory / "topics/index.json")
        require(index["topics"] == index["scenarios"], f"Inconsistent topic index: {platform}")
        require({entry["slug"] for entry in index["topics"]} == TOPICS,
                f"Unexpected topics: {platform}")
        counts = {"topics": 0, "posts": 0, "comments": 0, "user_instances": 0, "relationships": 0}

        for entry in index["topics"]:
            slug = entry["slug"]
            label = f"{platform}/{slug}"
            topic = directory / "topics" / slug
            meta = read(topic / "meta.json")
            data = read(topic / "env_data.json")
            users = read(topic / "UserAgent.json")
            read(topic / "recommendation.json")
            reference = directory / "doc/real_content_pool" / slug
            observed = read(reference / "content_pool_snapshot.json")
            read(reference / "user_comments.json")
            reference_meta = read(reference / "meta.json")
            require(meta["slug"] == slug, f"Wrong topic slug: {label}")
            require(entry["topic"] == meta["topic"] == data["scenario_topic"] == reference_meta["topic"],
                    f"Inconsistent topic title: {label}")
            posts = data["content_pool"]
            require(set(posts) == set(observed), f"Seed/reference post mismatch: {label}")
            require(len(posts) == entry["post_count"] == meta["post_count"], f"Post count mismatch: {label}")
            require(len(users) == entry["user_count"] == meta["portrait"]["UserAgent"],
                    f"User count mismatch: {label}")
            for identifier, post in posts.items():
                require(str(post["id"]) == identifier, f"Post ID mismatch: {label}")
                require(not post.get("comments"), f"Initial simulation comments must be empty: {label}")
            for identifier, user in users.items():
                require(str(user["id"]) == identifier, f"User ID mismatch: {label}")
            with available(topic / "Relationship.csv").open(encoding="utf-8-sig", newline="") as stream:
                relationships = list(csv.DictReader(stream))
            require(len(relationships) == entry["relationship_count"] == meta["relationship_count"],
                    f"Relationship count mismatch: {label}")
            for relationship in relationships:
                require(relationship["source_agent"] in users and relationship["target_agent"] in users,
                        f"Relationship endpoint missing from profiles: {label}")
            comment_count = sum(len(post.get("comments", [])) for post in observed.values())
            require(comment_count == entry["comment_count"] == meta["comment_count"],
                    f"Observed comment count mismatch: {label}")

            relative = data["reference_embedding_path"]
            vector_path = (root / relative).resolve()
            require(vector_path.is_relative_to(root), f"Reference path outside checkout: {label}")
            if relative not in vector_cache:
                vectors = read(vector_path)
                ids = set()
                for vector in vectors:
                    identifier = str(vector.get("post_id") or vector.get("note_id"))
                    require(identifier not in ids, f"Duplicate reference vector ID: {relative}")
                    ids.add(identifier)
                    values = vector["embedding"]
                    require(len(values) == 768 and all(isinstance(v, (float, int)) and math.isfinite(v) for v in values)
                            and any(values), f"Invalid reference vector: {relative}/{identifier}")
                vector_cache[relative] = ids
                report["reference_vectors"][relative] = len(ids)
            needs_vector = {identifier for identifier, post in observed.items() if post.get("comments")}
            require(needs_vector <= vector_cache[relative],
                    f"Missing reference vectors: {label}: {sorted(needs_vector - vector_cache[relative])}")
            counts["topics"] += 1
            counts["posts"] += len(posts)
            counts["comments"] += comment_count
            counts["user_instances"] += len(users)
            counts["relationships"] += len(relationships)

        files = ("posts.jsonl", "comments.jsonl", "user_profiles.jsonl", "relationships.csv",
                 "historical_posts.jsonl", "historical_comments.jsonl") if dataset == "x" else (
                     "posts.csv", "comments.csv", "profiles.csv", "relationships.csv",
                     "historical_posts.csv", "historical_post_comments.csv")
        for filename in files:
            available(root / "datasets" / dataset / filename)
        report["platforms"][platform] = counts

    for filename in ("__init__.py", "event.py", "eventbus.py", "scheduler.py"):
        available(root / "src/onesim/events" / filename)
    for filename in ("app.py", "main.py"):
        ast.parse(available(root / "src" / filename).read_text(encoding="utf-8-sig"))

    if publication:
        require({p.name for p in config.iterdir()} == CONFIGS, "Publication must contain exactly seven config files")
        require({p.name for p in (root / "datasets").iterdir()} == set(DATASETS), "Unexpected dataset directory")
        require({p.name for p in (root / "src/envs").iterdir()} == set(PLATFORMS) | {".gitignore"},
                "Unexpected environment directory")
        allowed = {".git", ".gitignore", ".gitattributes", "README.md", "LICENSE", "setup.py",
                   "src", "config", "datasets", "docs", "scripts"}
        require({p.name for p in root.iterdir()} <= allowed, "Unexpected publication root entry")
        for entries in models.values():
            for model in entries:
                require(model.get("api_key") in {"EMPTY", "YOUR_API_KEY"}, "Publication model key must be a placeholder")
        for path in (root / "src").rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Checkout to inspect")
    parser.add_argument("--publication", action="store_true", help="Also check the clean publication directory")
    args = parser.parse_args()
    try:
        report = check(args.root, args.publication)
    except (ValueError, KeyError, TypeError, OSError, SyntaxError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("PASS: " + json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
