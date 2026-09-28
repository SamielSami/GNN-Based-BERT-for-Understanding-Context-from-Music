"""Training-only label-prior, ontology-name keyword and TF-IDF controls."""
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline


class LabelPrior:
    def __init__(self, targets):
        self.priors = targets.mean(axis=0)

    def predict_proba(self, texts):
        return np.tile(self.priors, (len(texts), 1))


class OntologyKeywords:
    """Fixed official display-name phrases, no caption-derived target generation.

    Comma-separated aliases and parenthetical qualifiers are normalized. A
    trailing 'music' is also removed for genre phrases. No dataset-fitted rules.
    """
    def __init__(self, tags):
        names = json.loads(Path(__file__).with_name("audioset_names.json").read_text(encoding="utf-8"))
        self.patterns = []
        for tag in tags:
            phrases = []
            for alias in names.get(tag, tag).lower().split(","):
                phrase = re.sub(r"\s*\([^)]*\)", "", alias).strip()
                phrases.append(phrase)
                if phrase.endswith(" music"):
                    phrases.append(phrase[:-6])
            self.patterns.append(re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.I))

    def predict_proba(self, texts):
        return np.array([[float(bool(p.search(text))) for p in self.patterns] for text in texts])


def fit_baselines(texts, labels, tags, splits, seed):
    from .train import select_thresholds
    train_texts = [texts[i] for i in splits["train"]]
    val_texts = [texts[i] for i in splits["validation"]]
    train_y, val_y = labels[splits["train"]], labels[splits["validation"]]
    tfidf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000, sublinear_tf=True),
        OneVsRestClassifier(LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=seed)),
    )
    tfidf.fit(train_texts, train_y)
    models = {"label_prior": LabelPrior(train_y), "ontology_keyword": OntologyKeywords(tags),
              "tfidf_logistic": tfidf}
    return {name: (model, select_thresholds(val_y, model.predict_proba(val_texts)))
            for name, model in models.items()}


def evaluate_baselines(models, texts, labels, tags, splits, frame, output_dir):
    from .train import classification_report
    reports = {}
    for name, (model, calibration) in models.items():
        reports[name] = {"calibration": calibration}
        for split in ("validation", "test"):
            indices = splits[split]
            probabilities = model.predict_proba([texts[i] for i in indices])
            thresholds = np.asarray(calibration["thresholds"])
            reports[name][split] = classification_report(labels[indices], probabilities, tags, thresholds)
            np.savez_compressed(output_dir / f"{name}_{split}_predictions.npz",
                                ytid=frame.iloc[indices].ytid.to_numpy(dtype=str),
                                row_indices=np.asarray(indices), tags=np.asarray(tags), targets=labels[indices],
                                probabilities=probabilities, thresholds=thresholds)
        joblib.dump({"model": model, "calibration": calibration, "tags": tags}, output_dir / f"{name}.joblib")
    return reports
