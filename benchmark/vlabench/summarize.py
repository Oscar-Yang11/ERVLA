import os
import sys
import json
import argparse
import pandas as pd

metric_map = {
    "success_rate": "SR",
    "progress_score": "PS",
    "intention_score": "IS",
}


def _ensure_dir(p: str):
    d = os.path.dirname(os.path.abspath(p))
    if d and (not os.path.exists(d)):
        os.makedirs(d, exist_ok=True)


def visulize_step_bar(df, outfile):
    import matplotlib.pyplot as plt
    import numpy as np

    df_plot = df.reset_index(drop=True) if isinstance(df.index, pd.MultiIndex) else df
    if df_plot.empty:
        print(f"[visulize_step_bar] df is empty, skip plotting {outfile}")
        return
    if "track" not in df_plot.columns or "model" not in df_plot.columns:
        print(f"[visulize_step_bar] Missing 'track' or 'model' column, columns = {df_plot.columns}")
        return

    metric_alphas = [1.0, 0.50, 0.20]  # SR deepest, PS mid, IS lightest
    tracks = sorted(df_plot["track"].unique())
    models = sorted(df_plot["model"].unique())
    if len(models) == 0 or len(tracks) == 0:
        print(f"[visulize_step_bar] No models ({len(models)}) or tracks ({len(tracks)}), skip plotting {outfile}")
        return

    x = np.arange(len(tracks))
    n_models = len(models)
    total_width = 0.8
    bar_width = total_width / n_models
    color_palette = plt.get_cmap("tab10").colors

    plt.figure(figsize=(max(7, 1.8 * len(tracks)), 6))
    ax = plt.gca()

    for m_idx, model in enumerate(models):
        color = color_palette[m_idx % len(color_palette)]
        bar_positions = x - total_width / 2 + m_idx * bar_width + bar_width / 2

        sr_vals, ps_vals, is_vals = [], [], []
        for track in tracks:
            _row = df_plot[(df_plot["track"] == track) & (df_plot["model"] == model)]
            sr = _row["avg_SR"].values[0] if len(_row["avg_SR"]) else 0.0
            ps = _row["avg_PS"].values[0] if len(_row["avg_PS"]) else 0.0
            is_ = _row["avg_IS"].values[0] if len(_row["avg_IS"]) else 0.0
            sr_vals.append(sr)
            ps_vals.append(ps)
            is_vals.append(is_)

        # IS base
        ax.bar(
            bar_positions,
            is_vals,
            width=bar_width,
            color=color,
            alpha=metric_alphas[2],
            label=model,
            edgecolor="black",
            linewidth=0.5,
        )
        # (PS - SR) stacked on SR (kept your original style)
        ax.bar(
            bar_positions,
            [ps - sr for ps, sr in zip(ps_vals, sr_vals)],
            width=bar_width,
            bottom=sr_vals,
            color=color,
            alpha=metric_alphas[1],
            edgecolor="black",
            linewidth=0.5,
            label=None,
        )
        # SR top
        ax.bar(
            bar_positions,
            sr_vals,
            width=bar_width,
            color=color,
            alpha=metric_alphas[0],
            edgecolor="black",
            linewidth=0.5,
            label=None,
        )

    ax.set_xlabel("Track")
    ax.set_ylabel("Score")
    ax.set_title("IS/PS/SR")
    ax.set_xticks(x)
    ax.set_xticklabels(tracks, rotation=30)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    _ensure_dir(outfile)
    plt.savefig(outfile, dpi=180)
    print(f"Saved figure: {outfile}")


def visulize(df, outfile):
    import matplotlib.pyplot as plt
    import numpy as np

    df_plot = df.reset_index(drop=True) if isinstance(df.index, pd.MultiIndex) else df
    if df_plot.empty:
        print(f"[visulize] df is empty, skip plotting {outfile}")
        return
    if "track" not in df_plot.columns or "model" not in df_plot.columns:
        print(f"[visulize] Missing 'track' or 'model' column, columns = {df_plot.columns}")
        return

    metrics = ["avg_SR", "avg_PS", "avg_IS"]
    metric_names = {"avg_SR": "SR", "avg_PS": "PS", "avg_IS": "IS"}
    tracks = sorted(df_plot["track"].unique())
    models = sorted(df_plot["model"].unique())
    if len(models) == 0 or len(tracks) == 0:
        print(f"[visulize] No models ({len(models)}) or tracks ({len(tracks)}), skip plotting {outfile}")
        return

    x = np.arange(len(tracks))
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6), sharey=True)
    if n_metrics == 1:
        axes = [axes]

    total_width = 0.8
    width = total_width / len(models)

    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx]
        for i, model in enumerate(models):
            values = []
            for track in tracks:
                v = df_plot[(df_plot["track"] == track) & (df_plot["model"] == model)][metric]
                v = v.values[0] if len(v) else float("nan")
                values.append(v)
            pos = x + (i - (len(models) - 1) / 2) * width
            ax.bar(pos, values, width=width, label=model)

        ax.set_xlabel("Track")
        ax.set_ylabel("Score")
        ax.set_title(f"{metric_names[metric]} on each track")
        ax.set_xticks(x)
        ax.set_xticklabels(tracks, rotation=30)
        if m_idx == n_metrics - 1:
            ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    _ensure_dir(outfile)
    plt.savefig(outfile, dpi=180)
    print(f"Saved figure: {outfile}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        required=True,
        type=str,
        help="Root dir that contains track_* subfolders, each with metrics.json",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Optional model name shown in table/figures. Default: basename(root_dir).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Output directory. Default: same as root_dir.",
    )
    parser.add_argument(
        "--out_name",
        default="vlabench_summary.xlsx",
        type=str,
        help="Excel filename (written under out_dir).",
    )
    parser.add_argument(
        "--fig_name",
        default="vlabench_figure.png",
        type=str,
        help="Figure filename (written under out_dir).",
    )
    parser.add_argument(
        "--fig_stack_name",
        default="vlabench_figure_stack.png",
        type=str,
        help="Stack figure filename (written under out_dir).",
    )
    parser.add_argument(
        "--md_name",
        default="vlabench_table.md",
        type=str,
        help="Markdown table filename (written under out_dir).",
    )

    args = parser.parse_args()
    root_dir = os.path.abspath(args.root_dir)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else root_dir

    outfile = os.path.join(out_dir, args.out_name)
    figure = os.path.join(out_dir, args.fig_name)
    figure_stack = os.path.join(out_dir, args.fig_stack_name)
    mdfile = os.path.join(out_dir, args.md_name)

    records = []
    task_list = set()

    if not os.path.isdir(root_dir):
        print(f"ERROR: root_dir not a directory: {root_dir}")
        sys.exit(1)

    # scan tracks
    for track in os.listdir(root_dir):
        track_dir = os.path.join(root_dir, track)
        if not os.path.isdir(track_dir):
            continue
        if not track.startswith("track_"):
            continue

        metric_file = os.path.join(track_dir, "metrics.json")
        if not os.path.isfile(metric_file):
            continue

        with open(metric_file, "r") as f:
            metric_data = json.load(f)

        model_name = args.model_name if args.model_name else os.path.basename(root_dir)
        row = {"model": model_name, "track": track}

        sum_success, sum_intention, sum_progress, n = 0.0, 0.0, 0.0, 0
        # metrics.json: {task_name: {success_rate, progress_score, intention_score, ...}}
        for task_name, scores in metric_data.items():
            task_list.add(task_name)

            for k in ["success_rate", "progress_score", "intention_score"]:
                value = scores.get(k, None)
                if value is not None:
                    try:
                        value = round(float(value), 3)
                    except Exception:
                        pass
                row[f"{task_name}_{metric_map[k]}"] = value

            # avg
            sum_success += float(scores.get("success_rate", 0) or 0)
            sum_intention += float(scores.get("intention_score", 0) or 0)
            sum_progress += float(scores.get("progress_score", 0) or 0)
            n += 1

        row["avg_SR"] = round(sum_success / n, 3) if n > 0 else None
        row["avg_IS"] = round(sum_intention / n, 3) if n > 0 else None
        row["avg_PS"] = round(sum_progress / n, 3) if n > 0 else None
        records.append(row)

    print("Total records:", len(records))
    if len(records) == 0:
        print("ERROR: 没有找到任何 metrics.json，请检查 root_dir 是否设置正确。")
        sys.exit(0)

    task_list = sorted(task_list)

    # columns for excel / summary df
    columns = ["model", "track"]
    tuples = [("模型名", ""), ("评测track", "")]

    for task in task_list:
        for k in ["success_rate", "progress_score", "intention_score"]:
            col = f"{task}_{metric_map[k]}"
            columns.append(col)
            tuples.append((task, metric_map[k]))

    columns += ["avg_SR", "avg_IS", "avg_PS"]
    tuples += [("平均分", "SR"), ("平均分", "IS"), ("平均分", "PS")]

    df_raw = pd.DataFrame(records)
    for col in columns:
        if col not in df_raw.columns:
            df_raw[col] = None
    df_raw = df_raw[columns].sort_values(by=["model", "track"])

    # ===== Markdown: Track rows, task SR/PS cols + Avg SR/PS =====
    md_cols = ["track"]
    md_headers = ["Track"]

    for task in task_list:
        md_cols.append(f"{task}_SR")
        md_cols.append(f"{task}_PS")
        md_headers.append(f"{task} SR")
        md_headers.append(f"{task} PS")

    md_cols.extend(["avg_SR", "avg_PS"])
    md_headers.extend(["Avg SR", "Avg PS"])

    for c in md_cols:
        if c not in df_raw.columns:
            df_raw[c] = None
    md_df = df_raw[md_cols].copy()

    def _fmt(v):
        if v is None:
            return ""
        try:
            if pd.isna(v):
                return ""
        except TypeError:
            pass
        try:
            return f"{float(v):.3f}"
        except Exception:
            return str(v)

    for c in md_cols:
        if c == "track":
            continue
        md_df[c] = md_df[c].map(_fmt)

    _ensure_dir(mdfile)
    with open(mdfile, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(md_headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(md_headers)) + " |\n")
        for _, row in md_df.iterrows():
            values = [str(row["track"])]
            for c in md_cols[1:]:
                values.append(str(row[c]))
            f.write("| " + " | ".join(values) + " |\n")

    print(f"Saved markdown table: {mdfile}")

    # ===== Figures =====
    df_for_plot = df_raw.copy()
    visulize(df_for_plot, figure)
    visulize_step_bar(df_for_plot, figure_stack)

    # ===== Excel with MultiIndex columns =====
    df_excel = df_raw.copy()
    df_excel.index = pd.MultiIndex.from_arrays([df_excel["model"], df_excel["track"]], names=["model", "track"])
    df_excel = df_excel.drop(columns=["model", "track"])

    assert len(df_excel.columns) == len(tuples) - 2
    df_excel.columns = pd.MultiIndex.from_tuples(tuples[2:])

    _ensure_dir(outfile)
    writer = pd.ExcelWriter(outfile, engine="xlsxwriter")
    df_excel.to_excel(writer, sheet_name="summary")
    writer.close()
    print(f"Saved excel: {outfile}")


if __name__ == "__main__":
    main()
