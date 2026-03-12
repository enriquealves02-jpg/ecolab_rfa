import ast, os, sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config_v2 as config
matplotlib.rcParams["font.family"] = "DejaVu Sans"


def clean_label(rep, n=5):
    try:
        words = ast.literal_eval(rep)
        return "  *  ".join(str(w).strip() for w in words[:n] if str(w).strip())
    except Exception:
        s = str(rep)
        return "_".join(s.split("_")[1:]).replace("_", " ").strip() if "_" in s else s


def run():
    print("PIPELINE V2 ETAPE 3 - Visualisation topics")
    os.makedirs(config.OUT_DIR, exist_ok=True)

    # Lire topics_info_v2.csv depuis le pipeline v1
    import pathlib
    info_csv = pathlib.Path(config.PIPELINE_DIR) / "outputs" / "bertopic_v2" / "topics_info_v2.csv"
    if not info_csv.exists():
        print(f"WARN: topics_info_v2.csv introuvable : {info_csv}"); return

    df = pd.read_csv(info_csv)
    df = df[df["Topic"] != -1].sort_values("Count", ascending=True).reset_index(drop=True)
    df["label"] = df["Representation"].map(clean_label)

    # Plot 1 : distribution des topics
    n = len(df)
    fig, ax = plt.subplots(figsize=(13, max(8, n * 0.38)))
    bars = ax.barh(df["label"], df["Count"], color="#2c7bb6", height=0.7, edgecolor="white", linewidth=0.4)
    for bar, cnt in zip(bars, df["Count"]):
        ax.text(bar.get_width() + df["Count"].max() * 0.008, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}", va="center", ha="left", fontsize=7.5, color="#333")
    ax.set_xlabel("Nombre de documents", fontsize=11, labelpad=8)
    ax.set_title(f"BERTopic v2 - Distribution des {n} topics (hors outliers)", fontsize=12, fontweight="bold", pad=12)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="y", labelsize=7); ax.tick_params(axis="x", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, df["Count"].max() * 1.12)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out1 = os.path.join(config.OUT_DIR, "topics_distribution_v2.png")
    fig.savefig(out1, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  -> {out1}")

    # Plot 2 : scores si dispo
    if not os.path.exists(config.TOPIC_SCORE_SUMMARY_CSV):
        print("  (score plot ignore - lance etape 1 d'abord)"); return

    summary = pd.read_csv(config.TOPIC_SCORE_SUMMARY_CSV)
    d = summary.merge(df[["Topic", "label"]].rename(columns={"Topic": "topic"}), on="topic", how="left")
    d["label"] = d["label"].fillna(d["topic"].astype(str))
    d = d.sort_values("mean_score", ascending=True).reset_index(drop=True)

    def color(s):
        if s <= -0.5: return "#d73027"
        if s <   0:   return "#fc8d59"
        if s == 0:    return "#d9d9d9"
        if s <   0.5: return "#91cf60"
        return              "#1a9641"

    fig2, ax2 = plt.subplots(figsize=(14, max(8, len(d) * 0.38)))
    bars2 = ax2.barh(d["label"], d["mean_score"], color=[color(s) for s in d["mean_score"]],
                     height=0.72, edgecolor="white", linewidth=0.3)
    for bar, sc, nd in zip(bars2, d["mean_score"], d["n_docs"]):
        x = bar.get_width()
        ax2.text(x + (0.03 if x >= 0 else -0.03), bar.get_y() + bar.get_height() / 2,
                 f"{sc:+.2f} (n={nd})", va="center", ha="left" if x >= 0 else "right", fontsize=7.5)
    ax2.axvline(0, color="#555", linewidth=0.8)
    ax2.axvspan(-2, -0.1, alpha=0.04, color="#d73027")
    ax2.axvspan( 0.1,  2, alpha=0.04, color="#1a9641")
    ax2.set_xlabel("Score moyen Ollama  (-2 protecteur <-> +2 amenageur)", fontsize=10, labelpad=8)
    ax2.set_title(f"Pipeline v2 - Orientation des {len(d)} topics", fontsize=12, fontweight="bold", pad=12)
    ax2.tick_params(axis="y", labelsize=7); ax2.set_xlim(-2.4, 2.4)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(facecolor="#d73027", label="Tres protecteur (<=-.5)"),
        Patch(facecolor="#fc8d59", label="Protecteur (-.5 a 0)"),
        Patch(facecolor="#d9d9d9", label="Neutre"),
        Patch(facecolor="#91cf60", label="Amenageur (0 a +.5)"),
        Patch(facecolor="#1a9641", label="Tres amenageur (>=+.5)"),
    ], loc="lower right", fontsize=8)
    plt.tight_layout()
    out2 = os.path.join(config.OUT_DIR, "topics_scores_v2.png")
    fig2.savefig(out2, dpi=160, bbox_inches="tight"); plt.close(fig2)
    print(f"  -> {out2}")
    print("ETAPE 3 TERMINEE")


if __name__ == "__main__":
    run()
