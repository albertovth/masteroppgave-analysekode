


from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer



NO_STOPWORDS = {
    "a","aa","absolutt","ad","ade","alle","allerede","all","alt","altså","alltid","and","andre","annen","annet","ans",
    "at","av","bak","bare","begge","begge to","bakom","både","båe","bort","borte","bra","bruke","bruker","brukt",
    "bør","burde","ca","cirka","da","dag","dager","deg","de","dei","deim","deira","deires","dem","den","denne",
    "der","deres","dermed","dersom","det","dette","di","din","ditt","dine","disse","dvs","du","dykk","dykkar","då",
    "efter","eg","ein","eit","eitt","eller","ellers","en","endå","enn","er","etc","et","etter","faktisk","få","får",
    "fått","fantes","finn","finnes","finne","fant","feil","fleire","flere","flest","for","fordi","for eksempel",
    "forutan","fram","frå","fra","frem","fremme","før","føre","først","første","ga","ganske","ge","gi","gikk",
    "gjennom","gjorde","gjort","gjør","god","gode","godt","gonger","gå","gått","ha","hadde","har","hatt","hei","heile",
    "hele","heller","helst","hende","henne","henner","her","hjå","ho","hos","hva","hver","hvert","hvilken","hvilket",
    "hvilke","hvis","hvor","hvordan","hvorfor","i","ifølge","igjen","ikkje","ikke","in","innen","inn","ja","jeg","jo",
    "kan","kanskje","kjem","kom","kommet","kor","kordan","korleis","korso","kva","kvar","kvifor","kven","kun","kunne",
    "kvart","la","lag","lenger","lengre","like","likevel","litt","lok","lukk","låg","lå","lågt","låt","mai","man",
    "mange","med","meg","meget","meir","mellom","men","mens","mer","mest","mi","min","mitt","mine","må","måtte","nær",
    "nærme","når","nå","no","noko","noen","nok","nokså","nord","ny","nye","nytt","og","også","om","omtrent","oss",
    "over","part","pluss","på","rundt","sa","sagt","samme","samt","sjøl","skal","skulle","slik","slike","slutt","snart",
    "som","spesielt","statt","stod","stå","stått","så","sånn","sånne","særleg","særlig","ta","takk","tar","til","tilbake",
    "til dels","tillegg","til slutt","to","tok","topp","tre","treng","trenger","trengt","tror","under","unna","unntatt",
    "ut","uten","utan","utover","var","vart","ved","vel","verken","vi","via","vil","ville","viss","visst","vår","vårt",
    "våre","y","ye","yes","yo","you","ytterligere","åt","åtte","ø","øk","økt","ønske","ønsker","ønsket","å","år","års",
    "åpenbart","årlig","å være",
    # vanlige ord på norsk
    "ai","ki","kunstig","kunstige","intelligens","organisasjon","virksomhet","departement","departementet","etaten",
}

TOKEN_RE = re.compile(r"[A-Za-zÆØÅæøå]+", re.UNICODE)


def simple_tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def preprocess_series(
    s: pd.Series,
    stopwords: set[str],
    min_len: int = 3,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Returns:
      - cleaned docs as strings (tokens joined by space)
      - per-row stats (no raw text)
    """
    cleaned_docs: List[str] = []
    stats_rows = []

    for i, x in enumerate(s.fillna("").astype(str).tolist()):
        raw = x.strip()
        toks = [t for t in simple_tokenize(raw) if len(t) >= min_len and t not in stopwords]
        cleaned_docs.append(" ".join(toks))

        stats_rows.append(
            {
                "row_id": int(i),              # 0-basert rad-id
                "excel_row": int(i + 2),        # rad 1 er tittel -> første svar på rad 2
                "num_chars": int(len(raw)),
                "num_tokens": int(len(simple_tokenize(raw))),
                "num_tokens_after_clean": int(len(toks)),
                "is_empty_after_clean": bool(len(toks) == 0),
            }
        )

    return cleaned_docs, pd.DataFrame(stats_rows)


def fit_lda(
    docs: List[str],
    n_topics: int,
    max_features: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
    random_state: int = 42,
) -> Tuple[LatentDirichletAllocation, CountVectorizer, np.ndarray]:
    n_docs = len(docs)
    safe_min_df = int(min_df)
    if n_docs <= 1:
        safe_min_df = 1
    else:
        safe_min_df = min(safe_min_df, n_docs)

    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, int(ngram_max)),
        min_df=safe_min_df,
        max_df=float(max_df),
    )

    X = vectorizer.fit_transform(docs)
    if X.shape[1] == 0:
        raise ValueError(
            "Empty vocabulary after cleaning/vectorizing. "
            "Try: lower --min_df, reduce stopwords, or set --ngram_max 1."
        )

    lda = LatentDirichletAllocation(
        n_components=int(n_topics),
        random_state=int(random_state),
        learning_method="batch",
        max_iter=50,
        evaluate_every=-1,
    )
    doc_topic = lda.fit_transform(X)  # (n_docs, n_topics)
    return lda, vectorizer, doc_topic


def topic_word_table(
    lda: LatentDirichletAllocation,
    vectorizer: CountVectorizer,
    topn: int,
) -> pd.DataFrame:
    feature_names = np.array(vectorizer.get_feature_names_out())
    rows = []

    for k, topic in enumerate(lda.components_):
        top_idx = np.argsort(topic)[::-1][: int(topn)]
        words = feature_names[top_idx]
        weights = topic[top_idx]

        # normalisert innad i tema-sannsynligheter for å gjøre det lesbart
        probs = weights / (weights.sum() if weights.sum() > 0 else 1.0)

        rows.append(
            {
                "topic_id": k,
                "top_words": ", ".join(words.tolist()),
                "top_weights": ", ".join([f"{w:.3f}" for w in weights.tolist()]),
                "top_probs": ", ".join([f"{p:.3f}" for p in probs.tolist()]),
            }
        )

    return pd.DataFrame(rows)


def doc_topic_table(doc_topic: np.ndarray, prefix: str) -> pd.DataFrame:
    df = pd.DataFrame(doc_topic, columns=[f"{prefix}topic_{i}" for i in range(doc_topic.shape[1])])
    df.insert(0, "row_id", np.arange(len(df)))
    df.insert(1, "excel_row", np.arange(len(df)) + 2)  # forutsetter at første rad er tittel
    df["dominant_topic"] = df.drop(columns=["row_id", "excel_row"]).values.argmax(axis=1)
    df["dominant_topic_weight"] = df.drop(columns=["row_id", "excel_row", "dominant_topic"]).values.max(axis=1)
    return df


def prevalence_table(doc_topics: pd.DataFrame) -> pd.DataFrame:
    g = (
        doc_topics.groupby("dominant_topic", as_index=False)
        .agg(
            n=("row_id", "count"),
            avg_dominant_weight=("dominant_topic_weight", "mean"),
            median_dominant_weight=("dominant_topic_weight", "median"),
        )
        .rename(columns={"dominant_topic": "topic_id"})
    )
    g["share"] = g["n"] / float(len(doc_topics)) if len(doc_topics) else 0.0
    return g.sort_values(["n", "avg_dominant_weight"], ascending=[False, False]).reset_index(drop=True)


def suggest_label_from_topwords(top_words: str, k: int = 4) -> str:
    words = [w.strip() for w in str(top_words).split(",") if w.strip()]
    return " / ".join(words[:k]) if words else "Topic"


def build_high_level_sheet(
    kind: str,
    topics_df: pd.DataFrame,
    doc_topics_df: pd.DataFrame,
    raw_text: pd.Series | None,
    examples_per_topic: int,
    excerpt_len: int,
) -> pd.DataFrame:
    prev = prevalence_table(doc_topics_df)

    base = prev.merge(topics_df[["topic_id", "top_words", "top_probs"]], on="topic_id", how="left")
    base["suggested_label"] = base["top_words"].fillna("").apply(lambda s: suggest_label_from_topwords(s, k=4))
    base.insert(0, "kind", kind)

    # eksempelkolonner
    for j in range(1, int(examples_per_topic) + 1):
        base[f"ex{j}_row_id"] = np.nan
        base[f"ex{j}_excel_row"] = np.nan
        base[f"ex{j}_weight"] = np.nan
        if raw_text is not None:
            base[f"ex{j}_excerpt"] = ""

    # fyll eksempler per tema med dominerende-emne tilhørighet
    for i, row in base.iterrows():
        k = int(row["topic_id"])
        sub = doc_topics_df.loc[doc_topics_df["dominant_topic"] == k].sort_values(
            "dominant_topic_weight", ascending=False
        )
        sub = sub.head(int(examples_per_topic))

        for j, (_, r) in enumerate(sub.iterrows(), start=1):
            rid = int(r["row_id"])
            base.at[i, f"ex{j}_row_id"] = rid
            base.at[i, f"ex{j}_excel_row"] = int(r.get("excel_row", rid + 2))
            base.at[i, f"ex{j}_weight"] = float(r["dominant_topic_weight"])
            if raw_text is not None:
                txt = str(raw_text.iloc[rid]) if rid < len(raw_text) else ""
                txt = txt.replace("\n", " ").strip()
                base.at[i, f"ex{j}_excerpt"] = (txt[: int(excerpt_len)] + "…") if len(txt) > int(excerpt_len) else txt

    # bedre kolonnerekkefølge
    cols_first = ["kind", "topic_id", "suggested_label", "share", "n", "avg_dominant_weight", "median_dominant_weight", "top_words", "top_probs"]
    rest = [c for c in base.columns if c not in cols_first]
    return base[cols_first + rest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="Fritekstsvar.xlsx")
    ap.add_argument("--sheet", default="Fritekstsvar")
    ap.add_argument("--risk_col", default=0, help="Column index (0-based) or column name for RISK (A=0)")
    ap.add_argument("--benefit_col", default=1, help="Column index (0-based) or column name for BENEFIT (B=1)")
    ap.add_argument("--topics", type=int, default=6, help="Number of topics per column")
    ap.add_argument("--topn", type=int, default=12, help="Top words per topic")
    ap.add_argument("--max_features", type=int, default=3000)
    ap.add_argument("--ngram_max", type=int, default=2, help="1=unigrams, 2=uni+bi, 3=uni+bi+tri")
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.95)
    ap.add_argument("--output", default="lda_output.xlsx")
    ap.add_argument("--include_text_in_output", action="store_true", help="Writes raw text to output (OFF by default)")
    ap.add_argument("--examples_per_topic", type=int, default=5, help="Examples per topic in high-level sheet")
    ap.add_argument("--excerpt_len", type=int, default=160, help="Chars per excerpt (only if include_text_in_output)")
    args = ap.parse_args()

    # Løs baner relative til mappen til dette skriptet (med mindre baner oppgis eksplisitt)
    base_dir = Path(__file__).resolve().parent

    in_path = Path(args.file)
    if not in_path.is_absolute():
        in_path = (base_dir / in_path).resolve()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    print(f"[CWD] {Path.cwd()}")
    print(f"[BASE] {base_dir}")
    print(f"[IN ] {in_path}")
    print(f"[OUT] {out_path}")

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_excel(in_path, sheet_name=args.sheet, engine="openpyxl")  # header=0 default

    # Kontroll. Personvern (Ikke print råtekst)
    print(f"[READ] shape={df.shape}")
    print(f"[READ] columns={df.columns.tolist()}")
    safe_preview = pd.DataFrame(
        {
            "col": df.columns[: min(10, df.shape[1])],
            "non_null": [int(df[c].notna().sum()) for c in df.columns[: min(10, df.shape[1])]],
        }
    )
    print("[READ] preview(non-null counts):")
    print(safe_preview.to_string(index=False))

    # Løs kolonner (indeks eller navn)
    def get_col(col_spec):
        if isinstance(col_spec, str) and col_spec.isdigit():
            col_spec = int(col_spec)
        if isinstance(col_spec, int):
            return df.iloc[:, col_spec]
        return df[col_spec]

    risk_s = get_col(args.risk_col).reset_index(drop=True)
    ben_s = get_col(args.benefit_col).reset_index(drop=True)

    # Før behandling
    risk_docs, risk_stats = preprocess_series(risk_s, NO_STOPWORDS)
    ben_docs, ben_stats = preprocess_series(ben_s, NO_STOPWORDS)

    print(f"[CLEAN] risk: empty_after_clean={risk_stats['is_empty_after_clean'].sum()} / {len(risk_stats)}")
    print(f"[CLEAN] benefit: empty_after_clean={ben_stats['is_empty_after_clean'].sum()} / {len(ben_stats)}")

    # Tilpass LDA separat
    lda_r, vec_r, dt_r = fit_lda(
        risk_docs, args.topics, args.max_features, args.ngram_max, args.min_df, args.max_df
    )
    lda_b, vec_b, dt_b = fit_lda(
        ben_docs, args.topics, args.max_features, args.ngram_max, args.min_df, args.max_df
    )

    topics_r = topic_word_table(lda_r, vec_r, args.topn)
    topics_b = topic_word_table(lda_b, vec_b, args.topn)

    doc_topics_r = doc_topic_table(dt_r, prefix="risk_")
    doc_topics_b = doc_topic_table(dt_b, prefix="benefit_")

    # Valgfritt. Inkluder råtekst output (av som standard, men kan skrus på)
    if args.include_text_in_output:
        doc_topics_r["risk_text"] = risk_s.fillna("").astype(str).values
        doc_topics_b["benefit_text"] = ben_s.fillna("").astype(str).values

    # Oppsummering på nøyt nivå, med valgfrie sitater
    risk_raw = risk_s.fillna("").astype(str) if args.include_text_in_output else None
    ben_raw = ben_s.fillna("").astype(str) if args.include_text_in_output else None

    risk_high = build_high_level_sheet(
        kind="risk",
        topics_df=topics_r,
        doc_topics_df=doc_topics_r,
        raw_text=risk_raw,
        examples_per_topic=args.examples_per_topic,
        excerpt_len=args.excerpt_len,
    )
    ben_high = build_high_level_sheet(
        kind="benefit",
        topics_df=topics_b,
        doc_topics_df=doc_topics_b,
        raw_text=ben_raw,
        examples_per_topic=args.examples_per_topic,
        excerpt_len=args.excerpt_len,
    )

    # Skriv Excel
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        topics_r.to_excel(xw, index=False, sheet_name="risk_topics")
        topics_b.to_excel(xw, index=False, sheet_name="benefit_topics")

        doc_topics_r.merge(risk_stats, on=["row_id", "excel_row"], how="left").to_excel(
            xw, index=False, sheet_name="risk_doc_topics"
        )
        doc_topics_b.merge(ben_stats, on=["row_id", "excel_row"], how="left").to_excel(
            xw, index=False, sheet_name="benefit_doc_topics"
        )

        risk_high.to_excel(xw, index=False, sheet_name="risk_high_level")
        ben_high.to_excel(xw, index=False, sheet_name="benefit_high_level")

        # Kort oppsummering
        summary = pd.DataFrame(
            {
                "what": ["risk", "benefit"],
                "n_rows": [len(risk_s), len(ben_s)],
                "topics": [args.topics, args.topics],
                "topn": [args.topn, args.topn],
                "max_features": [args.max_features, args.max_features],
                "ngram_max": [args.ngram_max, args.ngram_max],
                "min_df": [args.min_df, args.min_df],
                "max_df": [args.max_df, args.max_df],
                "include_text_in_output": [bool(args.include_text_in_output), bool(args.include_text_in_output)],
                "examples_per_topic": [args.examples_per_topic, args.examples_per_topic],
                "excerpt_len": [args.excerpt_len, args.excerpt_len],
            }
        )
        summary.to_excel(xw, index=False, sheet_name="summary")

    print(f"[WRITE] {out_path}")
    print("[WRITE] sheets: risk_topics, benefit_topics, risk_doc_topics, benefit_doc_topics, risk_high_level, benefit_high_level, summary")
    print(f"[DONE] Open '{out_path.name}'.")


if __name__ == "__main__":
    main()

