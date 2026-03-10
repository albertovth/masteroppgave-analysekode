from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="potensialet.csv")
    ap.add_argument("--column", default="KI_Potensialet")
    ap.add_argument("--sep", default=";")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--output", default="lda_output.txt")
    ap.add_argument("--topics", type=int, default=10)
    ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--examples", type=int, default=0, help="Antall eksempeltekster per tema. 0 = ingen.")
    ap.add_argument("--use_tfidf", action="store_true")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent

    in_path = Path(args.file)
    if not in_path.is_absolute():
        in_path = (base_dir / in_path).resolve()

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (base_dir / out_path).resolve()

    print(f"[IN ] {in_path}")
    print(f"[OUT] {out_path}")

    df = pd.read_csv(in_path, sep=args.sep, encoding=args.encoding)
    texts = df[args.column].dropna().astype(str)

    norwegian_stopwords = [
        "og","i","på","av","for","med","til","som","en","et","er",
        "det","den","de","vi","du","jeg","han","hun","man","har",
        "hadde","var","blir","ble","om","å","kan","skal",
        "at","så","men","eller","oss","dere","seg"
    ]

    if args.use_tfidf:
        vectorizer = TfidfVectorizer(
            stop_words=norwegian_stopwords,
            max_df=0.95,
            min_df=2
        )
    else:
        vectorizer = CountVectorizer(
            stop_words=norwegian_stopwords,
            max_df=0.95,
            min_df=2
        )

    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=args.topics,
        random_state=0,
        learning_method="batch",
        max_iter=20
    )
    lda.fit(X)

    with open(out_path, "w", encoding="utf-8") as f:
        words = vectorizer.get_feature_names_out()

        for idx, topic in enumerate(lda.components_):
            top_idx = topic.argsort()[-args.topn:][::-1]
            top_terms = [(words[i], round(float(topic[i]), 2)) for i in top_idx]
            f.write(f"Tema {idx+1}: {top_terms}\n")

        doc_topic = lda.transform(X)
        out_df = df.loc[texts.index].copy()
        out_df["dominant_topic"] = doc_topic.argmax(axis=1)
        out_df["topic_score"] = doc_topic.max(axis=1)

        if args.examples > 0:
            for t in range(args.topics):
                f.write(f"\n=== Typiske svar for Tema {t+1} ===\n")
                tema_docs = out_df[out_df["dominant_topic"] == t].sort_values("topic_score", ascending=False)
                for i, (text, score) in enumerate(zip(tema_docs[args.column], tema_docs["topic_score"]), start=1):
                    if i > args.examples:
                        break
                    f.write(f"{i}. ({score:.2f}) {text}\n")

    print(f"[WRITE] {out_path}")


if __name__ == "__main__":
    main()
