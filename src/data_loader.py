from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd


RAW_DIR = Path("data/raw/scapt_split")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def str_to_bool(x):
    if x is None:
        return None
    x = str(x).strip().lower()
    if x == "true":
        return 1
    if x == "false":
        return 0
    return None


def parse_scapt_xml(xml_path: Path, domain: str, split: str) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []

    for sentence in root.findall(".//sentence"):
        sentence_id = sentence.attrib.get("id")
        text_node = sentence.find("text")
        sentence_text = text_node.text.strip() if text_node is not None and text_node.text else None

        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is None:
            continue

        for aspect in aspect_terms.findall("aspectTerm"):
            term = aspect.attrib.get("term")
            polarity = aspect.attrib.get("polarity")
            from_idx = aspect.attrib.get("from")
            to_idx = aspect.attrib.get("to")
            implicit_value = aspect.attrib.get("implicit_sentiment")
            opinion_words = aspect.attrib.get("opinion_words")

            rows.append({
                "id": len(rows) + 1,
                "source_sentence_id": sentence_id,
                "domain": domain,
                "split": split,
                "sentence": sentence_text,
                "target": term,
                "from": int(from_idx) if from_idx not in [None, ""] else None,
                "to": int(to_idx) if to_idx not in [None, ""] else None,
                "polarity": polarity,
                "is_implicit": str_to_bool(implicit_value),
                "opinion_words": opinion_words,
                "source_file": xml_path.name,
            })

    return pd.DataFrame(rows)


def save_df(df: pd.DataFrame, out_path: Path):
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | shape={df.shape}")


def main():
    configs = [
        ("laptop", "train", RAW_DIR / "laptop" / "Laptops_Train_v2_Implicit_Labeled.xml"),
        ("laptop", "test", RAW_DIR / "laptop" / "Laptops_Test_Gold_Implicit_Labeled.xml"),
        ("restaurant", "train", RAW_DIR / "restaurant" / "Restaurants_Train_v2_Implicit_Labeled.xml"),
        ("restaurant", "test", RAW_DIR / "restaurant" / "Restaurants_Test_Gold_Implicit_Labeled.xml"),
    ]

    all_dfs = []

    for domain, split, path in configs:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        df = parse_scapt_xml(path, domain, split)
        save_df(df, PROCESSED_DIR / f"{domain}_{split}.csv")
        print(df.head(5).to_string())
        print()
        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)
    save_df(df_all, PROCESSED_DIR / "semeval14_scapt_all.csv")

    df_isa = df_all[df_all["is_implicit"] == 1].copy()
    save_df(df_isa, PROCESSED_DIR / "semeval14_scapt_isa_only.csv")

    valid_polarities = {"positive", "negative", "neutral"}
    df_clean = df_all[df_all["polarity"].isin(valid_polarities)].copy()
    df_clean = df_clean[df_clean["is_implicit"].isin([0, 1])].copy()
    df_clean["is_implicit"] = df_clean["is_implicit"].astype(int)

    save_df(df_clean, PROCESSED_DIR / "semeval14_scapt_all_clean.csv")
    save_df(df_clean[df_clean["is_implicit"] == 1].copy(), PROCESSED_DIR / "semeval14_scapt_isa_only_clean.csv")

    print("\nSummary (raw):")
    print(df_all.groupby(["domain", "split", "is_implicit"]).size())

    print("\nSummary (clean):")
    print(df_clean.groupby(["domain", "split", "is_implicit"]).size())
    print("\nPolarity counts (clean):")
    print(df_clean["polarity"].value_counts())


if __name__ == "__main__":
    main()