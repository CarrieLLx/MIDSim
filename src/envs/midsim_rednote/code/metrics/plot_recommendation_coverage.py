#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 comments.csv 与仿真输出 datasets/step_* 快照，计算推荐覆盖指标并绘图。

示例：
  python plot_recommendation_coverage.py \\
    --datasets-root /path/to/run/datasets \\
    --comments-csv /path/to/YuLan-OneSim/datasets/openreview/comments.csv \\
    --out-png recommendation_coverage.png \\
    --out-csv recommendation_coverage.csv
"""
from __future__ import annotations

import argparse
import os
import sys

_metrics_dir = os.path.dirname(os.path.abspath(__file__))
if _metrics_dir not in sys.path:
    sys.path.insert(0, _metrics_dir)

from recommendation_coverage_metric import (  # noqa: E402
    run_recommendation_coverage_over_steps,
    run_login_validity_miss_over_steps,
    rows_to_csv,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="推荐覆盖指标曲线（相对有效样本数）")
    ap.add_argument(
        "--datasets-root",
        required=True,
        help="含 step_1, step_2, ... 的目录（一般为某次 run 的 datasets 路径）",
    )
    ap.add_argument(
        "--comments-csv",
        default=None,
        help="真实评论关系 CSV（默认：仓库 datasets/openreview/comments.csv）",
    )
    ap.add_argument("--out-png", default="recommendation_coverage.png", help="输出折线图路径（v1）")
    ap.add_argument("--out-csv", default=None, help="可选：输出每步指标 CSV（v1）")
    ap.add_argument(
        "--login-validity",
        action="store_true",
        help="额外计算：帖子 time≤仿真时刻且 last_login=本轮 timestamp；未出现在任意渠道=1；并画递推运行均值",
    )
    ap.add_argument(
        "--out-png-login",
        default="recommendation_coverage_login_validity.png",
        help="login-validity 模式折线图路径",
    )
    ap.add_argument("--out-csv-login", default=None, help="login-validity 每步指标 CSV")
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(_metrics_dir, "../../../../.."))
    comments_csv = args.comments_csv
    if not comments_csv:
        comments_csv = os.path.join(repo_root, "datasets", "openreview", "comments.csv")
    if not os.path.isfile(comments_csv):
        raise SystemExit(f"comments.csv 不存在: {comments_csv}")

    rows = run_recommendation_coverage_over_steps(args.datasets_root, comments_csv)
    if args.out_csv:
        rows_to_csv(rows, args.out_csv)

    if not rows and not args.login_validity:
        raise SystemExit("无有效 step 数据（检查 datasets-root 下是否有快照文件）")

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "需要 matplotlib 才能绘图：pip install matplotlib\n"
            f"已计算 v1 {len(rows)} 步，可改用 --out-csv 保存数值。原始错误: {e}"
        )

    if rows:

        def col(name: str) -> list:
            return [r.get(name) for r in rows]

        xs_raw = col("step_num")
        if xs_raw and all(x is not None for x in xs_raw):
            xs = xs_raw
        else:
            xs = list(range(1, len(rows) + 1))

        overall = col("overall_hit_ratio")
        ms = col("miss_social_ratio")
        mi = col("miss_interest_ratio")
        mr = col("miss_random_ratio")
        mh = col("miss_hot_ratio")
        neff = col("n_effective_pairs")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(xs, overall, marker="o", label="overall_hit_ratio（任一类推荐列表中出现过=1）")
        ax.plot(xs, ms, marker="s", label="miss_social_ratio（未在 social 列表=1）")
        ax.plot(xs, mi, marker="^", label="miss_interest_ratio（未在 interest 列表=1）")
        ax.plot(xs, mr, marker="v", label="miss_random_ratio（未在 random 列表=1）")
        ax.plot(xs, mh, marker="d", label="miss_hot_ratio（未在 hot 列表=1）")
        ax.set_xlabel("step（按 step_* 目录顺序）")
        ax.set_ylabel("比值（分子和 / N_effective）")
        ax.set_title("推荐覆盖 vs 有效样本（current_notes ∩ last_login>帖子time）")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(xs, neff, color="gray", linestyle="--", alpha=0.6, label="N_effective")
        ax2.set_ylabel("N_effective", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        fig.tight_layout()
        fig.savefig(args.out_png, dpi=150)
        print(f"Saved figure to {args.out_png}")
        if args.out_csv:
            print(f"Saved CSV to {args.out_csv}")

    if args.login_validity:
        rows_l = run_login_validity_miss_over_steps(args.datasets_root, comments_csv)
        if args.out_csv_login:
            rows_to_csv(rows_l, args.out_csv_login)
        if not rows_l:
            print("login-validity: 无有效 step（需 step_metadata.json 与用户快照）")
        else:
            xs_l = [r.get("step_num") for r in rows_l]
            if not all(x is not None for x in xs_l):
                xs_l = list(range(1, len(rows_l) + 1))

            def c2(name: str) -> list:
                return [r.get(name) for r in rows_l]

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.plot(xs_l, c2("miss_any_run_avg"), marker="o", label="miss_any_run_avg（未出现在任意渠道）")
            ax2.plot(xs_l, c2("miss_social_run_avg"), marker="s", label="miss_social_run_avg")
            ax2.plot(xs_l, c2("miss_interest_run_avg"), marker="^", label="miss_interest_run_avg")
            ax2.plot(xs_l, c2("miss_random_run_avg"), marker="v", label="miss_random_run_avg")
            ax2.plot(xs_l, c2("miss_hot_run_avg"), marker="d", label="miss_hot_run_avg")
            ax2.set_xlabel("step")
            ax2.set_ylabel("递推运行均值 R_n=(n-1)/n·R_{n-1}+x_n/n")
            ax2.set_title(
                "Login 对齐 + 帖子 time≤current_ts：未出现在渠道内=1 的递推均值"
            )
            ax2.legend(loc="best", fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2b = ax2.twinx()
            ax2b.plot(xs_l, c2("n_effective_login_validity"), color="gray", linestyle="--", alpha=0.6)
            ax2b.set_ylabel("N_effective", color="gray")
            fig2.tight_layout()
            fig2.savefig(args.out_png_login, dpi=150)
            print(f"Saved login-validity figure to {args.out_png_login}")
            if args.out_csv_login:
                print(f"Saved login-validity CSV to {args.out_csv_login}")


if __name__ == "__main__":
    main()
