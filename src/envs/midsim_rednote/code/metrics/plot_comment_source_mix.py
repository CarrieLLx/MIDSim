#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据最后一轮快照，统计全部评论在「算法推荐流 / 关注流 / 双渠道 / 其他」中的分布，并画饼图。

示例：
  python plot_comment_source_mix.py --datasets-root /path/to/run/datasets --out-png comment_mix.png
"""
from __future__ import annotations

import argparse
import os
import sys

_metrics_dir = os.path.dirname(os.path.abspath(__file__))
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)

from comment_source_mix import (  # noqa: E402
    count_comment_source_mix,
    load_final_snapshots_from_datasets_root,
    save_counts_json,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="评论来源饼图：算法推荐流 vs 关注流 vs 双渠道 vs 其他"
    )
    ap.add_argument(
        "--datasets-root",
        default=None,
        help="仿真 datasets 目录（默认使用最后一轮 step_*）",
    )
    ap.add_argument(
        "--content-pool-json",
        default=None,
        help="可选：直接指定 content_pool_snapshot.json 路径",
    )
    ap.add_argument(
        "--users-json",
        default=None,
        help="可选：直接指定 user_recommended_note_ids_by_channel.json",
    )
    ap.add_argument("--out-png", default="comment_source_mix.png", help="饼图输出路径")
    ap.add_argument("--out-json", default=None, help="可选：输出统计 JSON")
    args = ap.parse_args()

    if args.content_pool_json and args.users_json:
        import json

        with open(args.content_pool_json, "r", encoding="utf-8") as f:
            cp = json.load(f)
        with open(args.users_json, "r", encoding="utf-8") as f:
            us = json.load(f)
        step_label = "custom"
        if not isinstance(cp, dict):
            cp = {}
        if not isinstance(us, dict):
            us = {}
    elif args.datasets_root:
        loaded = load_final_snapshots_from_datasets_root(args.datasets_root)
        if loaded is None:
            raise SystemExit(
                f"无法加载最后一轮快照，请检查: {args.datasets_root} 下是否存在 "
                "step_*/content_pool_snapshot.json 与 user_recommended_note_ids_by_channel.json"
            )
        cp, us, step_label = loaded
    else:
        raise SystemExit("请提供 --datasets-root，或同时提供 --content-pool-json 与 --users-json")

    summary = count_comment_source_mix(cp, us)
    summary["step_dir_used"] = step_label

    if args.out_json:
        save_counts_json(summary, args.out_json)
        print(f"Saved JSON to {args.out_json}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(f"需要 matplotlib: pip install matplotlib\n{e}")

    c = summary["counts"]
    labels_map = [
        ("algo_only", "仅算法推荐流"),
        ("social_only", "仅关注流"),
        ("both", "双渠道（算法+关注）"),
        ("neither", "两流均未出现该帖"),
        ("unknown_user", "无用户快照"),
    ]
    sizes = [c.get(k, 0) for k, _ in labels_map]
    labels = [f"{lab}\n({n})" for (_, lab), n in zip(labels_map, sizes)]
    total = summary["total_comments"]
    if total == 0:
        raise SystemExit("content_pool 中无评论，无法作图")

    colors = ["#6BAED6", "#74C476", "#FD8D3C", "#CCCCCC", "#E377C2"]
    explode = (0.02, 0.02, 0.05, 0, 0)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        colors=colors[: len(sizes)],
        explode=explode[: len(sizes)],
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title(
        f"评论来源构成（共 {total} 条）\n依据用户 recommended_note_ids_by_channel\nstep: {step_label}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved pie chart to {args.out_png}")
    print("Counts:", c)
    print("Ratios:", summary["ratios"])


if __name__ == "__main__":
    main()
