"""
analysis_utils.py
Evaluation, vocabulary and LIME helpers for the LSTM sentiment model.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

CLASS_NAMES = ['Negative', 'Neutral', 'Positive']
CLASS_IDS = [0, 1, 2]
CLASS_COLORS = {'Negative': '#c0392b', 'Neutral': '#7f8c8d', 'Positive': '#27ae60'}
CLASS_CMAPS = {'Negative': 'Reds', 'Neutral': 'Greys', 'Positive': 'Greens'}

# The four metrics the assignment asks for.
METRIC_KEYS = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Stop words worth keeping since they flip sentiment.
SENTIMENT_STOPWORDS_TO_KEEP = frozenset({
    'no', 'not', 'nor', 'never', 'none', 'nothing', 'nowhere', 'cannot',
    'without', 'but', 'however', 'although', 'though', 'very', 'too',
    'only', 'again', 'against', 'off', 'least', 'less', 'enough',
})


def make_stopword_list(keep=SENTIMENT_STOPWORDS_TO_KEEP, verbose=True):
    """Sklearn's stop-word list minus the negation words we want to keep."""
    custom = frozenset(ENGLISH_STOP_WORDS - set(keep))
    if verbose:
        rescued = sorted(ENGLISH_STOP_WORDS & set(keep))
        print(f"sklearn stop words       : {len(ENGLISH_STOP_WORDS)}")
        print(f"Sentiment tokens rescued : {len(rescued)} -> {rescued}")
        print(f"Final stop-word list     : {len(custom)}")
    return list(custom)


# EVALUATION

def compute_metrics(y_true, y_pred, class_ids=CLASS_IDS):
    """Accuracy plus macro precision, recall and F1 as a flat dict."""
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_ids, average='macro', zero_division=0)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': prec,
        'recall_macro': rec,
        'f1_macro': f1,
    }


def metrics_table(y_true, y_pred, class_names=CLASS_NAMES, class_ids=CLASS_IDS,
                  save_csv=None):
    """Per-class precision/recall/F1 table. Returns (dataframe, summary dict)."""
    report = classification_report(
        y_true, y_pred, labels=class_ids, target_names=class_names,
        output_dict=True, zero_division=0)

    df = pd.DataFrame(report).T.loc[list(class_names) + ['macro avg', 'weighted avg']]
    df[['precision', 'recall', 'f1-score']] = df[['precision', 'recall', 'f1-score']].round(4)
    df['support'] = df['support'].astype(int)

    summary = compute_metrics(y_true, y_pred, class_ids)
    print(f"Accuracy: {summary['accuracy']:.4f} | Macro F1: {summary['f1_macro']:.4f}\n")

    if save_csv:
        df.to_csv(save_csv)
    return df, summary


def plot_training_curves(history, title='Training Dynamics', n_classes=3):
    """Loss and accuracy per epoch. `history` needs train/val loss and acc lists."""
    tr_loss, va_loss = history['train_loss'], history['val_loss']
    tr_acc, va_acc = history['train_acc'], history['val_acc']
    epochs = range(1, len(tr_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, tr_loss, marker='o', label='Train')
    axes[0].plot(epochs, va_loss, marker='s', label='Validation')
    axes[0].set_title('Cross-Entropy Loss per Epoch')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, tr_acc, marker='o', label='Train')
    axes[1].plot(epochs, va_acc, marker='s', label='Validation')
    axes[1].axhline(1 / n_classes, ls='--', c='grey', lw=1,
                    label=f'Random baseline ({1/n_classes:.3f})')
    axes[1].set_title('Accuracy per Epoch')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()

    gap = tr_acc[-1] - va_acc[-1]
    print(f"Final train acc {tr_acc[-1]:.4f} | val acc {va_acc[-1]:.4f} | gap {gap:+.4f}")
    print(f"Best val acc: {max(va_acc):.4f} at epoch {int(np.argmax(va_acc)) + 1}")


def plot_confusion_matrices(y_true, y_pred, class_names=CLASS_NAMES,
                            class_ids=CLASS_IDS, title_suffix=''):
    """Raw-count and row-normalised confusion matrices side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=class_ids, display_labels=class_names,
        cmap='Blues', colorbar=False, ax=axes[0], values_format='d')
    axes[0].set_title(f'Confusion Matrix — Counts{title_suffix}')

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=class_ids, display_labels=class_names,
        cmap='Blues', colorbar=False, ax=axes[1], normalize='true',
        values_format='.2f')
    axes[1].set_title(f'Confusion Matrix — Recall{title_suffix}')

    for ax in axes:
        ax.set_xlabel('Predicted sentiment'); ax.set_ylabel('True sentiment')
    plt.tight_layout(); plt.show()

    # Report each class's most common mistake.
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    for i, name in enumerate(class_names):
        wrong = {class_names[j]: cm_norm[i, j] for j in range(len(class_ids)) if j != i}
        worst = max(wrong, key=wrong.get)
        print(f"{name:<8}: recall {cm_norm[i, i]:.3f} | "
              f"most common error -> {worst} ({wrong[worst]:.3f})")
    return cm


def full_evaluation_report(y_true, y_pred, history=None, class_names=CLASS_NAMES,
                           class_ids=CLASS_IDS, model_label='Model', save_csv=None):
    """Runs the whole Step 4 evaluation: curves, metrics table and confusion matrices."""
    if history is not None:
        plot_training_curves(history, title=f'{model_label} Training Dynamics')

    print(f"\n===== {model_label}: Test-Set Metrics =====")
    df, summary = metrics_table(y_true, y_pred, class_names, class_ids, save_csv)
    print(df.to_string())

    print(f"\n===== {model_label}: Confusion =====")
    plot_confusion_matrices(y_true, y_pred, class_names, class_ids,
                            title_suffix=f' ({model_label})')
    return df, summary

# INFERENCE

def collect_probs_lstm(model, loader, device):
    """Run a DataLoader through the model and return (probs, labels)."""
    import torch
    import torch.nn.functional as F

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for inputs, lengths, y in loader:
            inputs = inputs.to(device)
            logits = model(inputs, lengths)   # lengths stay on CPU for packing
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
            labels.append(y.numpy())
    return np.vstack(probs), np.concatenate(labels)


def make_lstm_predict_proba(model, device, clean_and_tokenize, encode_and_truncate,
                            word2idx, max_length, batch_size=256):
    """Build the predict_proba(list[str]) callable LIME needs. Must mirror the
    notebook's preprocessing exactly, or LIME explains the wrong function."""
    import torch
    import torch.nn.functional as F
    from torch.nn.utils.rnn import pad_sequence

    def predict_proba(texts):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                seqs = [encode_and_truncate(clean_and_tokenize(t), word2idx, max_length)
                        for t in chunk]
                # encode_and_truncate handles the empty case, which happens because
                # LIME generates perturbations with every word deleted.
                lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
                padded = pad_sequence(seqs, batch_first=True, padding_value=0).to(device)
                logits = model(padded, lengths)
                out.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(out)

    return predict_proba


# VOCAB ANALYSIS

def build_class_bow(token_lists, labels, max_tokens=128, class_ids=CLASS_IDS,
                    stop_words=None, min_df=10, max_features=10000, verbose=True):
    """Build the document-term matrix and total the counts per sentiment class.
    Truncating to max_tokens keeps this to words the model actually sees."""
    if stop_words is None:
        stop_words = make_stopword_list(verbose=False)

    labels = np.asarray(labels)
    corpus = [' '.join(t[:max_tokens]) for t in token_lists]

    vec = CountVectorizer(
        stop_words=stop_words,
        token_pattern=r"(?u)\b[a-z]{2,}\b",   # tokens arrive already lowercased
        min_df=min_df,
        max_features=max_features,
    )
    dtm = vec.fit_transform(corpus)
    vocab = vec.get_feature_names_out()

    class_counts = np.vstack([
        np.asarray(dtm[labels == c].sum(axis=0)).ravel() for c in class_ids])
    total_counts = class_counts.sum(axis=0)

    if verbose:
        print(f"Documents: {dtm.shape[0]:,} | Vocabulary: {dtm.shape[1]:,}")
        print(f"Total tokens counted: {total_counts.sum():,}")

    return {
        'vectorizer': vec, 'dtm': dtm, 'vocab': vocab,
        'class_counts': class_counts, 'total_counts': total_counts,
        'w2col': {w: j for j, w in enumerate(vocab)},
    }


def log_odds_zscore(class_vec, total_vec, alpha0=1000.0):
    """Weighted log-odds of each word in one class vs the rest, with a Dirichlet
    prior so rare words don't dominate."""
    rest_vec = total_vec - class_vec
    n_c, n_r = class_vec.sum(), rest_vec.sum()

    prior = alpha0 * (total_vec / total_vec.sum())
    a0 = prior.sum()

    l_c = np.log((class_vec + prior) / (n_c + a0 - class_vec - prior))
    l_r = np.log((rest_vec + prior) / (n_r + a0 - rest_vec - prior))

    delta = l_c - l_r
    var = 1.0 / (class_vec + prior) + 1.0 / (rest_vec + prior)
    return delta / np.sqrt(var)


def compute_zscores(bow, class_ids=CLASS_IDS, alpha0=1000.0):
    """(C, V) matrix of log-odds z-scores, one row per class."""
    return np.vstack([log_odds_zscore(bow['class_counts'][i], bow['total_counts'], alpha0)
                      for i in range(len(class_ids))])


def top_words_table(bow, scores, top_n=20, class_names=CLASS_NAMES, label='score'):
    """Ranked table of the top words for each class, side by side."""
    vocab = bow['vocab']
    fmt = (lambda v: f"{v:,}") if label == 'count' else (lambda v: f"{v:.1f}")
    tbl = pd.DataFrame({
        name: [f"{vocab[j]} ({fmt(scores[i, j])})" for j in np.argsort(-scores[i])[:top_n]]
        for i, name in enumerate(class_names)
    }, index=[f"#{k+1}" for k in range(top_n)])

    # Words in every list are generic, not class-specific.
    top_sets = [set(vocab[np.argsort(-scores[i])[:top_n]]) for i in range(len(class_names))]
    shared = set.intersection(*top_sets)
    print(f"Words appearing in ALL {len(class_names)} top-{top_n} lists: "
          f"{len(shared)}/{top_n} -> {sorted(shared)}")
    return tbl


def plot_wordclouds(bow, scores, class_names=CLASS_NAMES, cmaps=CLASS_CMAPS,
                    subtitle='Raw Frequency', max_words=90, top_k=200,
                    random_state=498, suptitle=None):
    """One word cloud per sentiment class. Negative scores are dropped since
    clouds can only weight positive values."""
    from wordcloud import WordCloud

    vocab = bow['vocab']
    fig, axes = plt.subplots(1, len(class_names), figsize=(6.3 * len(class_names), 5))
    for i, name in enumerate(class_names):
        order = np.argsort(-scores[i])[:top_k]
        freqs = {vocab[j]: float(scores[i, j]) for j in order if scores[i, j] > 0}
        wc = WordCloud(width=800, height=440, background_color='white',
                       colormap=cmaps[name], max_words=max_words,
                       random_state=random_state,
                       prefer_horizontal=0.95).generate_from_frequencies(freqs)
        axes[i].imshow(wc, interpolation='bilinear')
        axes[i].axis('off')
        axes[i].set_title(f'{name} Reviews — {subtitle}', fontsize=13, pad=10)

    if suptitle:
        plt.suptitle(suptitle, fontsize=15, y=1.04)
    plt.tight_layout(); plt.show()


def plot_distinctive_words(bow, zscores, k=15, class_names=CLASS_NAMES,
                           colors=CLASS_COLORS):
    """Horizontal bars of the most class-characteristic words."""
    vocab = bow['vocab']
    fig, axes = plt.subplots(1, len(class_names), figsize=(5.7 * len(class_names), 6.5))
    for i, name in enumerate(class_names):
        order = np.argsort(-zscores[i])[:k][::-1]   # strongest ends up on top
        axes[i].barh([vocab[j] for j in order], [zscores[i, j] for j in order],
                     color=colors[name], alpha=0.85)
        axes[i].set_title(name, fontsize=12)
        axes[i].set_xlabel('Weighted log-odds z-score')
        axes[i].grid(axis='x', alpha=0.3)
    plt.suptitle(f'Top {k} Distinctive Words per Sentiment Class', fontsize=14, y=1.01)
    plt.tight_layout(); plt.show()


# LIME

def make_lime_explainer(class_names=CLASS_NAMES, random_state=498):
    """Construct a LimeTextExplainer for the three sentiment classes."""
    from lime.lime_text import LimeTextExplainer
    return LimeTextExplainer(class_names=list(class_names), random_state=random_state)


def plot_lime_instance(text, predict_proba, explainer, model_view=None, true_id=None,
                       class_names=CLASS_NAMES, class_ids=CLASS_IDS,
                       num_features=12, num_samples=1500):
    """Explain one review for all three classes and plot the word weights.
    Blue pushes toward the class, red pushes away."""
    shown = model_view(text) if model_view else text
    exp = explainer.explain_instance(
        shown, predict_proba,
        num_features=num_features, num_samples=num_samples,
        labels=list(class_ids),
    )
    probs = predict_proba([shown])[0]

    fig, axes = plt.subplots(1, len(class_names), figsize=(5.7 * len(class_names), 5.2))
    for k, name in enumerate(class_names):
        pairs = exp.as_list(label=class_ids[k])
        words = [w for w, _ in pairs][::-1]
        wts = [v for _, v in pairs][::-1]
        colors = ['#2980b9' if v > 0 else '#c0392b' for v in wts]
        axes[k].barh(words, wts, color=colors, alpha=0.85)
        axes[k].axvline(0, color='black', lw=0.8)
        axes[k].set_title(f'{name}  (p = {probs[k]:.3f})', fontsize=12)
        axes[k].set_xlabel('LIME weight')
        axes[k].grid(axis='x', alpha=0.3)

    hdr = f'Predicted: {class_names[int(probs.argmax())]}'
    if true_id is not None:
        hdr += f'  |  Actual: {class_names[true_id]}'
    plt.suptitle(f'LIME — per-class word contributions   ({hdr})\n'
                 'blue = pushes TOWARD this class, red = pushes AWAY',
                 fontsize=13, y=1.06)
    plt.tight_layout(); plt.show()
    return exp


def run_lime_batch(texts, labels, predict_proba, explainer, model_view=None,
                   n_per_class=30, num_samples=800, num_features=25,
                   class_names=CLASS_NAMES, class_ids=CLASS_IDS,
                   random_state=498, save_csv=None, log_every=10):
    """Run LIME on a stratified sample of reviews and collect every
    (word, class, weight) triple into a tidy DataFrame."""
    texts = list(texts)
    labels = np.asarray(labels)
    rng = np.random.default_rng(random_state)

    sample_idx = np.concatenate([
        rng.choice(np.where(labels == c)[0],
                   size=min(n_per_class, int((labels == c).sum())), replace=False)
        for c in class_ids])

    print(f"Explaining {len(sample_idx)} reviews x {num_samples} perturbations "
          f"= {len(sample_idx) * num_samples:,} forward passes...")

    records = []
    t0 = time.time()
    for n, idx in enumerate(sample_idx, 1):
        shown = model_view(texts[int(idx)]) if model_view else texts[int(idx)]
        exp = explainer.explain_instance(
            shown, predict_proba, num_features=num_features,
            num_samples=num_samples, labels=list(class_ids))
        for k, cid in enumerate(class_ids):
            for word, weight in exp.as_list(label=cid):
                records.append({'doc': int(idx), 'word': word.lower(),
                                'class': class_names[k], 'weight': weight,
                                'true_class': class_names[int(labels[idx])]})
        if log_every and n % log_every == 0:
            rate = n / (time.time() - t0)
            print(f"  {n}/{len(sample_idx)} reviews  ({rate:.2f}/s, "
                  f"~{(len(sample_idx) - n) / rate / 60:.1f} min left)", flush=True)

    df = pd.DataFrame(records)
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. {len(df):,} (word, class) "
          f"weights over {df.word.nunique():,} unique words.")
    if save_csv:
        df.to_csv(save_csv, index=False)
    return df


def aggregate_lime(lime_df, min_obs=5, class_names=CLASS_NAMES, verbose=True):
    """Average each word's LIME weight across every review it appeared in.
    Returns (influence, obs_count, agg)."""
    agg = (lime_df.groupby(['word', 'class'])['weight']
           .agg(mean_weight='mean', std_weight='std', n_docs='size')
           .reset_index())

    influence = agg.pivot(index='word', columns='class', values='mean_weight')[list(class_names)]
    obs_count = agg.pivot(index='word', columns='class', values='n_docs')[list(class_names)]

    # fillna(0) matters: LIME picks features per label, so a word can be missing
    # for one class and a bare .min() would skip the NaN and let it through.
    keep = obs_count.fillna(0).min(axis=1) >= min_obs
    influence = influence[keep].fillna(0.0)

    if verbose:
        print(f"Words retained (seen in >= {min_obs} explanations for every class): "
              f"{len(influence)}")
        for name in class_names:
            top = influence[name].sort_values(ascending=False).head(12)
            print(f"\nPushes most strongly TOWARD {name}:")
            print('  ' + ', '.join(f'{w} ({v:+.3f})' for w, v in top.items()))

    return influence, obs_count, agg


def plot_influence_heatmap(influence, bow, top_freq=30, class_names=CLASS_NAMES,
                           save_csv=None):
    """Heatmap of mean LIME influence for the most frequent BoW words, i.e.
    whether the common vocabulary is what actually drives predictions."""
    vocab, total_counts = bow['vocab'], bow['total_counts']
    freq_order = [vocab[j] for j in np.argsort(-total_counts)]
    freq_words = [w for w in freq_order if w in influence.index][:top_freq]

    heat = influence.loc[freq_words]
    lim = np.abs(heat.values).max()

    fig, ax = plt.subplots(figsize=(7.5, max(5, 0.36 * len(freq_words))))
    im = ax.imshow(heat.values, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(range(len(freq_words))); ax.set_yticklabels(freq_words, fontsize=9)
    for i in range(len(freq_words)):
        for j in range(len(class_names)):
            v = heat.values[i, j]
            ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=7.5,
                    color='white' if abs(v) > lim * 0.55 else 'black')
    ax.set_title(f'Mean LIME Influence of the {top_freq} Most Frequent Words\n'
                 'red = pushes toward this class,  blue = pushes away',
                 fontsize=12, pad=12)
    plt.colorbar(im, ax=ax, shrink=0.4, label='Mean LIME weight')
    plt.tight_layout(); plt.show()

    if save_csv:
        heat.round(4).to_csv(save_csv)
    return heat
