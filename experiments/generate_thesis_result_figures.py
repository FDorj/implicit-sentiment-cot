from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


PERSIAN_DIGIT_TRANSLATION = str.maketrans("0123456789.-", "۰۱۲۳۴۵۶۷۸۹٫−")


def _format_persian_number(value: int | float, precision: int | None = None) -> str:
    if precision is None:
        rendered = str(int(value)) if isinstance(value, int) or float(value).is_integer() else str(value)
    else:
        rendered = f"{float(value):.{precision}f}"
    return rendered.translate(PERSIAN_DIGIT_TRANSLATION)


THESIS_FONT_DIR = (
    Path(__file__).resolve().parents[1]
    / "قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir"
    / "Fonts"
)


MAIN_METHOD_ORDER = [
    "TF-IDF + Logistic Regression",
    "Direct Qwen3 8B",
    "THOR simplified",
    "THOR original-ish SC3",
    "Simple reflection",
    "ETC standard",
    "ETC over original-ish SC3",
    "Final selected pipeline",
]

LABEL_ORDER = ["negative", "neutral", "positive"]

FIGURE_STEMS = [
    "ch5_qwen_full_test_methods",
    "ch5_direct_vs_final_confusion",
    "ch5_selector_behavior",
    "ch5_qwen_gemini_shared_subset",
    "ch5_gemini_direct_vs_selected_confusion",
]

KEY_COLUMNS = [
    "id",
    "source_sentence_id",
    "sentence",
    "target",
    "from",
    "to",
    "domain",
    "split",
]


def _results_path(repo_root: Path, filename: str) -> Path:
    path = Path(repo_root) / "results" / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _test_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "split" not in frame.columns:
        raise ValueError(f"Missing split column in {path}")
    test = frame.loc[frame["split"].eq("test")].copy()
    if test.empty:
        raise ValueError(f"No test rows in {path}")
    return test


def _with_stable_index(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    missing = [column for column in KEY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing stable-key columns in {path}: {missing}")

    indexed = frame.copy()
    for column in KEY_COLUMNS:
        indexed[column] = indexed[column].fillna("").astype(str)
    indexed = indexed.set_index(KEY_COLUMNS).sort_index()
    if not indexed.index.is_unique:
        raise ValueError(f"Duplicate stable sample keys in {path}")
    return indexed


def load_main_test_results(repo_root: Path) -> list[dict]:
    path = _results_path(repo_root, "final_results_table.csv")
    frame = pd.read_csv(path)
    test = frame.loc[frame["split"].eq("test")].copy()
    test = test.set_index("method")

    missing = [method for method in MAIN_METHOD_ORDER if method not in test.index]
    if missing:
        raise ValueError(f"Missing main test methods: {missing}")

    rows = []
    for method in MAIN_METHOD_ORDER:
        row = test.loc[method]
        if int(row["n_eval"]) != 442:
            raise ValueError(f"Unexpected test size for {method}: {row['n_eval']}")
        rows.append(
            {
                "method": method,
                "n_eval": int(row["n_eval"]),
                "accuracy": float(row["accuracy"]),
                "macro_f1": float(row["macro_f1"]),
            }
        )
    return rows


def _load_aligned_direct_final(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct_path = _results_path(repo_root, "direct_isa_predictions.csv")
    final_path = _results_path(
        repo_root, "etc_thor_originalish_sc3_guarded_tuned_selected_isa_predictions.csv"
    )
    direct = _with_stable_index(_test_rows(direct_path), direct_path)
    final = _with_stable_index(_test_rows(final_path), final_path)

    if len(direct) != 442 or len(final) != 442:
        raise ValueError(f"Expected 442 test rows, got direct={len(direct)}, final={len(final)}")
    if not direct.index.equals(final.index):
        raise ValueError("Direct and final test sample keys are not aligned")
    if not direct["polarity"].astype(str).equals(final["polarity"].astype(str)):
        raise ValueError("Direct and final gold labels differ")
    if not direct["prediction"].astype(str).equals(final["direct_prediction"].astype(str)):
        raise ValueError("Saved final direct predictions do not match the direct result file")
    return direct, final


def load_confusion_data(repo_root: Path) -> dict:
    direct, final = _load_aligned_direct_final(repo_root)
    gold = final["polarity"]
    direct_counts = confusion_matrix(
        gold, direct["prediction"], labels=LABEL_ORDER
    ).tolist()
    final_counts = confusion_matrix(
        gold, final["selected_prediction"], labels=LABEL_ORDER
    ).tolist()
    return {
        "n": len(final),
        "labels": LABEL_ORDER.copy(),
        "direct_counts": direct_counts,
        "final_counts": final_counts,
    }


def load_gemini_confusion_data(repo_root: Path) -> dict:
    direct_path = _results_path(repo_root, "gemini_modelcmp_direct_subset_predictions.csv")
    final_path = _results_path(
        repo_root, "gemini_modelcmp_validation_tuned_selected_predictions.csv"
    )
    direct = _with_stable_index(_test_rows(direct_path), direct_path)
    final = _with_stable_index(_test_rows(final_path), final_path)

    if len(direct) != 90 or len(final) != 90:
        raise ValueError(
            f"Expected 90 shared Gemini test rows, got direct={len(direct)}, final={len(final)}"
        )
    if not direct.index.equals(final.index):
        raise ValueError("Gemini Direct and selected-profile sample keys are not aligned")
    if not direct["polarity"].astype(str).equals(final["polarity"].astype(str)):
        raise ValueError("Gemini Direct and selected-profile gold labels differ")
    if not direct["prediction"].astype(str).equals(final["direct_prediction"].astype(str)):
        raise ValueError("Saved Gemini selected-profile Direct labels do not match Direct output")

    return {
        "n": len(final),
        "labels": LABEL_ORDER.copy(),
        "direct_counts": confusion_matrix(
            final["polarity"], direct["prediction"], labels=LABEL_ORDER
        ).tolist(),
        "final_counts": confusion_matrix(
            final["polarity"], final["selected_prediction"], labels=LABEL_ORDER
        ).tolist(),
    }


def load_selector_behavior(repo_root: Path) -> dict:
    _, final = _load_aligned_direct_final(repo_root)
    counts = final["selected_source"].value_counts().to_dict()
    source_counts = {
        source: int(counts.get(source, 0))
        for source in ["direct", "thor", "diagnostic"]
    }

    direct_correct = final["direct_prediction"].eq(final["polarity"])
    final_correct = final["selected_prediction"].eq(final["polarity"])
    transitions = {
        "both_correct": int((direct_correct & final_correct).sum()),
        "gain": int((~direct_correct & final_correct).sum()),
        "loss": int((direct_correct & ~final_correct).sum()),
        "both_wrong": int((~direct_correct & ~final_correct).sum()),
    }
    return {"n": len(final), "source_counts": source_counts, "transitions": transitions}


def _selected_test_macro_f1(path: Path) -> tuple[int, float]:
    frame = _test_rows(path)
    required = {"polarity", "selected_prediction"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing selected prediction columns in {path}: {missing}")
    score = f1_score(frame["polarity"], frame["selected_prediction"], average="macro")
    return len(frame), float(score)


def load_shared_subset_comparison(repo_root: Path) -> dict:
    summary_path = _results_path(repo_root, "gemini_qwen_modelcmp_subset_summary.csv")
    summary = pd.read_csv(summary_path)
    test = summary.loc[summary["split"].eq("test")].set_index("method")

    method_rows = {
        "Direct": ("qwen_direct", "gemini_direct"),
        "THOR SC3": ("qwen_thor_sc3", "gemini_thor_sc3"),
    }
    values: dict[str, dict[str, float]] = {}
    sample_sizes = set()
    for label, (qwen_method, gemini_method) in method_rows.items():
        if qwen_method not in test.index or gemini_method not in test.index:
            raise ValueError(f"Missing shared-subset method rows for {label}")
        qwen_row = test.loc[qwen_method]
        gemini_row = test.loc[gemini_method]
        sample_sizes.update([int(qwen_row["n"]), int(gemini_row["n"])])
        values[label] = {
            "Qwen3 8B": float(qwen_row["macro_f1"]),
            "Gemini 2.5 Flash": float(gemini_row["macro_f1"]),
        }

    qwen_n, qwen_selected = _selected_test_macro_f1(
        _results_path(repo_root, "qwen_modelcmp_validation_tuned_selected_predictions.csv")
    )
    gemini_n, gemini_selected = _selected_test_macro_f1(
        _results_path(repo_root, "gemini_modelcmp_validation_tuned_selected_predictions.csv")
    )
    sample_sizes.update([qwen_n, gemini_n])
    if sample_sizes != {90}:
        raise ValueError(f"Shared test comparison must use n=90, got {sorted(sample_sizes)}")

    values["Validation-tuned selected"] = {
        "Qwen3 8B": qwen_selected,
        "Gemini 2.5 Flash": gemini_selected,
    }
    return {"n": 90, "macro_f1": values}


def _preamble(figure_id: str) -> str:
    font_path = THESIS_FONT_DIR.as_posix() + "/"
    return rf"""\documentclass[tikz,border=4pt]{{standalone}}
\usepackage{{pgfplots}}
\usetikzlibrary{{positioning,calc}}
\pgfplotsset{{compat=1.18}}
\definecolor{{PlotBlue}}{{HTML}}{{0072B2}}
\definecolor{{PlotOrange}}{{HTML}}{{E69F00}}
\definecolor{{PlotGreen}}{{HTML}}{{009E73}}
\definecolor{{PlotRed}}{{HTML}}{{D55E00}}
\definecolor{{PlotGray}}{{HTML}}{{7A7A7A}}
\usepackage{{xepersian}}
\settextfont[Path={{{font_path}}},BoldFont={{B Nazanin Bold.TTF}}]{{B Nazanin.TTF}}
\setlatintextfont[Path={{{font_path}}},BoldFont={{timesbd.ttf}}]{{times.ttf}}
\setdigitfont[Path={{{font_path}}},BoldFont={{Yas Bd.ttf}}]{{Yas.ttf}}
\newfontfamily\persiannumeralfont[Path={{{font_path}}},BoldFont={{Yas Bd.ttf}}]{{Yas.ttf}}
\begin{{document}}
% figure-id: {figure_id}
"""


def _document(body: str, figure_id: str) -> str:
    return _preamble(figure_id) + body + "\n\\end{document}\n"


def _format_coordinates(rows: list[dict], field: str, *, omit_final: bool) -> str:
    chosen = rows[:-1] if omit_final else rows[-1:]
    labels = {
        "TF-IDF + Logistic Regression": r"\lr{TF-IDF + Logistic Regression}",
        "Direct Qwen3 8B": r"پیش‌بینی مستقیم \lr{Qwen3 8B}",
        "THOR simplified": r"\lr{THOR} ساده‌شده",
        "Simple reflection": "بازبینی ساده",
        "ETC standard": r"کنترل‌گر \lr{ETC}",
        "THOR original-ish SC3": r"\lr{THOR SC3} سه‌اجرایی",
        "ETC over original-ish SC3": r"کنترل‌گر \lr{ETC} روی \lr{THOR SC3}",
        "Final selected pipeline": "سامانۀ نهایی",
    }
    coordinates = []
    for row in chosen:
        label = _format_persian_number(row[field], 3)
        coordinates.append(f"({row[field]:.6f},{{{labels[row['method']]}}}) [{label}]")
    return " ".join(coordinates)


def _render_main_results(rows: list[dict]) -> str:
    y_labels = (
        r"\lr{TF-IDF + Logistic Regression},پیش‌بینی مستقیم \lr{Qwen3 8B},"
        r"\lr{THOR} ساده‌شده,\lr{THOR SC3} سه‌اجرایی,بازبینی ساده,"
        r"کنترل‌گر \lr{ETC},کنترل‌گر \lr{ETC} روی \lr{THOR SC3},سامانۀ نهایی"
    )
    accuracy = _format_coordinates(rows, "accuracy", omit_final=True)
    macro_f1 = _format_coordinates(rows, "macro_f1", omit_final=True)
    final_accuracy = _format_coordinates(rows, "accuracy", omit_final=False)
    final_macro_f1 = _format_coordinates(rows, "macro_f1", omit_final=False)
    body = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=11.4cm,
  height=9.2cm,
  xbar,
  xmin=0.00,
  xmax=0.80,
  xtick={{0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80}},
  xticklabels={{۰,۰٫۱,۰٫۲,۰٫۳,۰٫۴,۰٫۵,۰٫۶,۰٫۷,۰٫۸}},
  xlabel={{امتیاز}},
  symbolic y coords={{{y_labels}}},
  ytick={{{y_labels}}},
  y dir=reverse,
  bar width=5pt,
  enlarge y limits=0.10,
  yticklabel style={{font=\small,align=right}},
  xticklabel style={{font=\persiannumeralfont\small}},
  tick label style={{font=\small}},
  label style={{font=\small}},
  grid=major,
  grid style={{draw=black!12}},
  axis line style={{draw=black!55}},
  legend style={{at={{(0.5,1.02)}},anchor=south,legend columns=2,draw=none,font=\small}},
  point meta=explicit symbolic,
  scatter/@pre marker code/.code={{}},
  scatter/@post marker code/.code={{}},
  nodes near coords*={{\pgfplotspointmeta}},
  every node near coord/.append style={{font=\persiannumeralfont\scriptsize,anchor=west,xshift=1pt}},
]
\addplot[draw=PlotBlue,fill=PlotBlue!45,bar shift=-3pt] coordinates {{{accuracy}}};
\addlegendentry{{دقت}}
\addplot[draw=PlotOrange,fill=PlotOrange!55,bar shift=3pt] coordinates {{{macro_f1}}};
\addlegendentry{{اف‌یک ماکرو}}
\addplot[draw=PlotBlue!80!black,fill=PlotBlue,bar shift=-3pt,forget plot] coordinates {{{final_accuracy}}};
\addplot[draw=PlotOrange!80!black,fill=PlotOrange,bar shift=3pt,forget plot] coordinates {{{final_macro_f1}}};
\end{{axis}}
\end{{tikzpicture}}"""
    return _document(body, "qwen-full-test-methods")


def _matrix_panel(counts: list[list[int]], x_offset: float, title: str) -> str:
    display_labels = ["منفی", "خنثی", "مثبت"]
    lines = [rf"\node[font=\bfseries] at ({x_offset + 2.1:.2f},4.35) {{{title}}};"]
    for column, label in enumerate(display_labels):
        lines.append(
            rf"\node[font=\small] at ({x_offset + column * 1.4 + 0.7:.2f},3.72) {{{label}}};"
        )
    for row, label in enumerate(display_labels):
        y = 2.4 - row * 1.05
        lines.append(rf"\node[font=\small,anchor=east] at ({x_offset - 0.12:.2f},{y + 0.525:.3f}) {{{label}}};")
        row_total = sum(counts[row])
        for column, count in enumerate(counts[row]):
            percentage = count / row_total
            count_label = _format_persian_number(count)
            percentage_label = _format_persian_number(percentage * 100, 1)
            shade = max(5, round(percentage * 100))
            text_color = "white" if percentage >= 0.56 else "black"
            x = x_offset + column * 1.4
            lines.append(
                rf"\filldraw[fill=PlotBlue!{shade},draw=white,line width=1pt] "
                rf"({x:.2f},{y:.2f}) rectangle ++(1.4,1.05);"
            )
            lines.append(
                rf"\node[text={text_color},font=\persiannumeralfont\small,align=center] at "
                rf"({x + 0.7:.2f},{y + 0.525:.3f}) "
                rf"{{\shortstack{{{count_label}\\{percentage_label}\%}}}};"
            )
    lines.append(rf"\node[font=\small] at ({x_offset + 2.1:.2f},-0.95) {{برچسب پیش‌بینی‌شده}};")
    lines.append(rf"\node[font=\small,rotate=90] at ({x_offset - 2.05:.2f},1.35) {{برچسب واقعی}};")
    return "\n".join(lines)


def _render_confusion_matrices(
    data: dict,
    *,
    figure_id: str = "direct-vs-final-confusion",
    left_title: str = r"پیش‌بینی مستقیم \lr{Qwen3 8B}",
    right_title: str = "سامانۀ نهایی",
    footer: str = "نرمال‌سازی سطری: هر خانه شمار و درصد سطر را نشان می‌دهد",
) -> str:
    left = _matrix_panel(data["direct_counts"], 0.0, left_title)
    right = _matrix_panel(data["final_counts"], 6.6, right_title)
    body = rf"""\begin{{tikzpicture}}
{left}
{right}
\node[font=\scriptsize,text=black!65] at (5.4,-1.55) {{{footer}}};
\end{{tikzpicture}}"""
    return _document(body, figure_id)


def _render_selector_behavior(data: dict) -> str:
    sources = data["source_counts"]
    transitions = data["transitions"]
    sample_count = _format_persian_number(data["n"])
    direct_count = _format_persian_number(sources["direct"])
    thor_count = _format_persian_number(sources["thor"])
    diagnostic_count = _format_persian_number(sources["diagnostic"])
    body = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  xbar,
  width=6.8cm,
  height=5.6cm,
  xmin=0,
  xmax=450,
  xtick={{0,50,100,150,200,250,300,350,400,450}},
  xticklabels={{۰,۵۰,۱۰۰,۱۵۰,۲۰۰,۲۵۰,۳۰۰,۳۵۰,۴۰۰,۴۵۰}},
  symbolic y coords={{مستقیم,\lr{{THOR}},تشخیصی}},
  ytick=data,
  y dir=reverse,
  xlabel={{تعداد نمونه‌های منتخب}},
  title={{\RL{{منبع منتخب در آزمون؛ تعداد نمونه‌ها: {sample_count}}}}},
  title style={{font=\small\bfseries}},
  tick label style={{font=\small}},
  xticklabel style={{font=\persiannumeralfont\small}},
  label style={{font=\small}},
  bar width=11pt,
  grid=major,
  grid style={{draw=black!12}},
  axis line style={{draw=black!55}},
  point meta=explicit symbolic,
  scatter/@pre marker code/.code={{}},
  scatter/@post marker code/.code={{}},
  nodes near coords*={{\pgfplotspointmeta}},
  every node near coord/.append style={{font=\persiannumeralfont\small,anchor=west,xshift=2pt}},
]
\addplot[draw=PlotBlue,fill=PlotBlue!70] coordinates {{({sources['direct']},مستقیم) [{direct_count}] ({sources['thor']},\lr{{THOR}}) [{thor_count}] ({sources['diagnostic']},تشخیصی) [{diagnostic_count}]}};
\end{{axis}}

\begin{{scope}}[xshift=9.0cm,yshift=0.25cm]
\node[font=\small\bfseries] at (2.2,4.75) {{\RL{{گذار وضعیت صحت؛ تعداد نمونه‌ها: {sample_count}}}}};
\node[font=\small] at (1.1,4.08) {{نهایی نادرست}};
\node[font=\small] at (3.3,4.08) {{نهایی درست}};
\node[font=\small,anchor=east] at (-0.15,3.05) {{مستقیم نادرست}};
\node[font=\small,anchor=east] at (-0.15,1.35) {{مستقیم درست}};

% هر دو نادرست
\filldraw[fill=PlotGray!22,draw=white,line width=1pt] (0,2.2) rectangle (2.2,3.9);
\node[font=\small,align=center] at (1.1,3.05) {{\shortstack{{هر دو نادرست\\\textbf{{{_format_persian_number(transitions['both_wrong'])}}}}}}};
% مستقیم نادرست / نهایی درست = بهبود
\filldraw[fill=PlotGreen!45,draw=white,line width=1pt] (2.2,2.2) rectangle (4.4,3.9);
\node[font=\small,align=center] at (3.3,3.05) {{\shortstack{{بهبود\\\textbf{{{_format_persian_number(transitions['gain'])}}}}}}};
% مستقیم درست / نهایی نادرست = تضعیف
\filldraw[fill=PlotRed!38,draw=white,line width=1pt] (0,0.5) rectangle (2.2,2.2);
\node[font=\small,align=center] at (1.1,1.35) {{\shortstack{{تضعیف\\\textbf{{{_format_persian_number(transitions['loss'])}}}}}}};
% هر دو درست
\filldraw[fill=PlotBlue!38,draw=white,line width=1pt] (2.2,0.5) rectangle (4.4,2.2);
\node[font=\small,align=center] at (3.3,1.35) {{\shortstack{{هر دو درست\\\textbf{{{_format_persian_number(transitions['both_correct'])}}}}}}};
\end{{scope}}
\end{{tikzpicture}}"""
    return _document(body, "selector-behavior")


def _render_shared_subset(data: dict) -> str:
    values = data["macro_f1"]
    qwen = " ".join(
        [
            f"(مستقیم,{values['Direct']['Qwen3 8B']:.6f}) [{_format_persian_number(values['Direct']['Qwen3 8B'], 3)}]",
            f"({{\\lr{{THOR SC3}}}},{values['THOR SC3']['Qwen3 8B']:.6f}) [{_format_persian_number(values['THOR SC3']['Qwen3 8B'], 3)}]",
            f"({{سیاست منتخب}},{values['Validation-tuned selected']['Qwen3 8B']:.6f}) [{_format_persian_number(values['Validation-tuned selected']['Qwen3 8B'], 3)}]",
        ]
    )
    gemini = " ".join(
        [
            f"(مستقیم,{values['Direct']['Gemini 2.5 Flash']:.6f}) [{_format_persian_number(values['Direct']['Gemini 2.5 Flash'], 3)}]",
            f"({{\\lr{{THOR SC3}}}},{values['THOR SC3']['Gemini 2.5 Flash']:.6f}) [{_format_persian_number(values['THOR SC3']['Gemini 2.5 Flash'], 3)}]",
            f"({{سیاست منتخب}},{values['Validation-tuned selected']['Gemini 2.5 Flash']:.6f}) [{_format_persian_number(values['Validation-tuned selected']['Gemini 2.5 Flash'], 3)}]",
        ]
    )
    body = rf"""\begin{{tikzpicture}}
\begin{{axis}}[
  width=11.5cm,
  height=7.2cm,
  ybar,
  ymin=0.00,
  ymax=0.90,
  ytick={{0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90}},
  yticklabels={{۰,۰٫۱,۰٫۲,۰٫۳,۰٫۴,۰٫۵,۰٫۶,۰٫۷,۰٫۸,۰٫۹}},
  ylabel={{اف‌یک ماکروی آزمون}},
  symbolic x coords={{مستقیم,\lr{{THOR SC3}},سیاست منتخب}},
  xtick=data,
  bar width=15pt,
  enlarge x limits=0.24,
  title={{\RL{{زیرمجموعۀ متوازن مشترک آزمون؛ تعداد نمونه‌ها: {_format_persian_number(data['n'])}}}}},
  title style={{at={{(axis description cs:0.5,1.16)}},anchor=south,font=\small\bfseries}},
  tick label style={{font=\small}},
  yticklabel style={{font=\persiannumeralfont\small}},
  label style={{font=\small}},
  grid=major,
  grid style={{draw=black!12}},
  axis line style={{draw=black!55}},
  legend style={{at={{(0.5,1.02)}},anchor=south,legend columns=2,draw=none,font=\small}},
  point meta=explicit symbolic,
  scatter/@pre marker code/.code={{}},
  scatter/@post marker code/.code={{}},
  nodes near coords*={{\pgfplotspointmeta}},
  every node near coord/.append style={{font=\persiannumeralfont\scriptsize,anchor=south,yshift=1pt}},
]
\addplot[draw=PlotBlue!80!black,fill=PlotBlue!75] coordinates {{{qwen}}};
\addlegendentry{{\lr{{Qwen3 8B}}}}
\addplot[draw=PlotOrange!80!black,fill=PlotOrange!80] coordinates {{{gemini}}};
\addlegendentry{{\lr{{Gemini 2.5 Flash}}}}
\end{{axis}}
\end{{tikzpicture}}"""
    return _document(body, "qwen-gemini-shared-subset")


def render_figure_tex(repo_root: Path, output_dir: Path) -> list[Path]:
    repo_root = Path(repo_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = {
        FIGURE_STEMS[0]: _render_main_results(load_main_test_results(repo_root)),
        FIGURE_STEMS[1]: _render_confusion_matrices(load_confusion_data(repo_root)),
        FIGURE_STEMS[2]: _render_selector_behavior(load_selector_behavior(repo_root)),
        FIGURE_STEMS[3]: _render_shared_subset(load_shared_subset_comparison(repo_root)),
        FIGURE_STEMS[4]: _render_confusion_matrices(
            load_gemini_confusion_data(repo_root),
            figure_id="gemini-direct-vs-selected-confusion",
            left_title=r"پیش‌بینی مستقیم \lr{Gemini 2.5 Flash}",
            right_title=r"سیاست منتخب \lr{Gemini}",
            footer=r"\RL{زیرمجموعۀ متوازن مشترک آزمون؛ تعداد نمونه‌ها: ۹۰؛ شمار و درصدهای نرمال‌شدۀ سطری}",
        ),
    }
    paths = []
    for stem in FIGURE_STEMS:
        path = output_dir / f"{stem}.tex"
        path.write_text(documents[stem], encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def _run_checked(command: list[str], cwd: Path, timeout: int = 120) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        output = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(command)}\n{output}"
        )


def _clean_known_build_files(build_dir: Path) -> None:
    for stem in FIGURE_STEMS:
        for suffix in [".tex", ".aux", ".log", ".out", ".pdf", ".png"]:
            path = build_dir / f"{stem}{suffix}"
            if path.is_file():
                path.unlink()


def build_figures(repo_root: Path, output_dir: Path) -> list[Path]:
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    xelatex = shutil.which("xelatex")
    pdftocairo = shutil.which("pdftocairo")
    if not xelatex:
        raise RuntimeError("xelatex is required to build thesis figures")
    if not pdftocairo:
        raise RuntimeError("pdftocairo is required to create PNG previews")

    build_dir = repo_root / ".test_tmp" / "thesis-figure-build"
    build_dir.mkdir(parents=True, exist_ok=True)
    _clean_known_build_files(build_dir)
    tex_paths = render_figure_tex(repo_root, build_dir)

    assets: list[Path] = []
    for tex_path in tex_paths:
        stem = tex_path.stem
        _run_checked(
            [
                xelatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-file-line-error",
                tex_path.name,
            ],
            cwd=build_dir,
        )
        built_pdf = build_dir / f"{stem}.pdf"
        if not built_pdf.is_file():
            raise RuntimeError(f"XeLaTeX did not create {built_pdf}")

        _run_checked(
            [
                pdftocairo,
                "-png",
                "-singlefile",
                "-r",
                "300",
                built_pdf.name,
                stem,
            ],
            cwd=build_dir,
        )
        built_png = build_dir / f"{stem}.png"
        if not built_png.is_file():
            raise RuntimeError(f"pdftocairo did not create {built_png}")

        final_pdf = output_dir / built_pdf.name
        final_png = output_dir / built_png.name
        shutil.copy2(built_pdf, final_pdf)
        shutil.copy2(built_png, final_png)
        assets.extend([final_pdf, final_png])

    for stem in FIGURE_STEMS:
        for suffix in [".aux", ".log", ".out"]:
            auxiliary = build_dir / f"{stem}{suffix}"
            if auxiliary.is_file():
                auxiliary.unlink()
    return assets


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (
        repo_root
        / "قالب__تمپلیت__پایان_نامه_امیرکبیر_thesis_template_of_Amirkabir"
        / "Images"
        / "Chapter5"
    )
    assets = build_figures(repo_root, output_dir)
    print(f"Generated {len(assets)} assets in {output_dir}")
    for path in assets:
        print(path)


if __name__ == "__main__":
    main()
