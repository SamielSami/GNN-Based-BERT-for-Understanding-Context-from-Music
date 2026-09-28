"""Build the real-data Task 4 review notebook; optionally execute every cell."""
import argparse
from pathlib import Path

import nbformat as nbf


def build():
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        md("""# Task 4 — measured contrastive graph–text retrieval

Review the completed three-seed experiment on 3,964 real MusicCaps pairs.
Trained DistilBERT and temporal GraphSAGE are frozen. Two small linear heads
project their cached representations into a normalized 64-dimensional space.
Symmetric InfoNCE uses matching IDs as positives and other batch pairs as negatives.

Default execution reads saved results and demonstrates validation retrieval;
it does not retrain or repeat final test inference. See `report/task4_results.md`
and `docs/task4/README.md` for full methods, limitations and reproduction."""),
        code("""from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / 'results/task4/available_run1'
assert (RUN / 'test_evaluation.json').exists(), 'Run the Task 4 suite first; see docs/task4/README.md.'
selection = json.loads((RUN / 'selection.json').read_text(encoding='utf-8'))
declaration = json.loads((RUN / 'suite_config.json').read_text(encoding='utf-8'))
verification = json.loads((RUN / 'verification.json').read_text(encoding='utf-8'))
assert not declaration['synthetic'] and verification['valid']
print('Real paired data | Seeds:', selection['seeds'])
print('Selection:', declaration['selection_metric'])
display(pd.DataFrame([verification]))"""),
        md("""## Bidirectional held-out comparison

Every seed uses the same 606 test queries and gallery. Recall is a fraction in
the CSV below. Mean and sample standard deviation summarize projection seeds,
not encoder variation or statistical confidence. Initial projections are the
exact pretraining weights for each seed. Random expected Recall@K is K/606.
CLAP is reported in the extension section; human listening remains pending."""),
        code("""comparison = pd.read_csv(RUN / 'comparison.csv')
aggregate = pd.read_csv(RUN / 'comparison_aggregate.csv')
display(aggregate[aggregate.split == 'test'])
display(Image(filename=str(RUN / 'analysis/retrieval_comparison.png')))
display(comparison[comparison.baseline == 'trained'])"""),
        md("""## Training and validation-only selection

Batch size 32, temperature 0.07, AdamW learning rate 0.001, weight decay 0.01,
at most 30 epochs, patience 5. Earliest maximum validation mean Recall@1 selects
each checkpoint. All seeds and checkpoint hashes were frozen before test inference.
Stars mark selected epochs; final training-loss minima do not select checkpoints."""),
        code("""display(pd.DataFrame([{k: v for k, v in r.items() if k not in {'artifacts', 'feature_sha256'}} for r in selection['runs']]))
display(Image(filename=str(RUN / 'analysis/learning_curves.png')))
for seed in selection['seeds']:
    display(Markdown(f'**Seed {seed}: loss and directional validation recall**'))
    display(Image(filename=str(RUN / f'seed{seed}/learning_curves.png')))"""),
        md("""## Independent metric recomputation

Recompute both trained and initial-baseline metrics directly from saved test
embeddings. No checkpoint inference occurs. File hashes connect these artifacts
to the frozen suite; IDs preserve row pairing."""),
        code("""from src.task3.data import feature_fingerprint
from src.task4.metrics import recompute
final = json.loads((RUN / 'test_evaluation.json').read_text(encoding='utf-8'))
assert feature_fingerprint(RUN / 'selection.json') == final['selection_sha256']
for seed in selection['seeds']:
    directory = RUN / f'seed{seed}/test'
    report = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    for filename, key in [('embeddings.npz', 'metrics'), ('untrained_embeddings.npz', 'untrained_projection_metrics')]:
        path = directory / filename
        assert feature_fingerprint(path) == final['artifacts'][path.relative_to(RUN).as_posix()]
        assert recompute(path) == report[key]
print('All six held-out embedding archives reproduce their bidirectional metrics.')"""),
        md("""## Retrieval demonstration from saved embeddings

Use the first declared seed and first validation query by default. Change
`DIRECTION` or `QUERY_INDEX` to inspect another cached query. Candidate captions
describe the associated tracks; no new raw text/audio is encoded in this cell.
Tied negatives rank ahead of the positive, matching aggregate metrics."""),
        code("""from src.task4.metrics import normalized_pairs, paired_ranks
DIRECTION = 'text_to_graph'  # or 'graph_to_text'
QUERY_INDEX = 0
seed = selection['seeds'][0]
with np.load(RUN / f'seed{seed}/validation/embeddings.npz', allow_pickle=False) as saved:
    texts, graphs = normalized_pairs(saved['text'], saved['graph'])
    ids = saved['sample_ids'].tolist()
queries, gallery = (texts, graphs) if DIRECTION == 'text_to_graph' else (graphs, texts)
assert 0 <= QUERY_INDEX < len(ids)
scores = queries[QUERY_INDEX] @ gallery.T
order = np.lexsort((np.arange(len(ids)) == QUERY_INDEX, -scores))[:10]
print('Direction:', DIRECTION, '| Query:', ids[QUERY_INDEX])
print('Paired rank:', int(paired_ranks(queries, gallery)[QUERY_INDEX]), '/', len(ids))
display(pd.DataFrame([{'rank': rank, 'sample_id': ids[i], 'cosine': float(scores[i]),
                       'paired_positive': bool(i == QUERY_INDEX)} for rank, i in enumerate(order, 1)]))"""),
        md("""## Examples, failures and duplicate captions

Fixed first-three queries and explicitly marked best/worst-rank diagnostics use
the first declared seed. These are caption-based observations, not listening
judgments. Distinct descriptions of similar music can be false negatives even
when exact/normalized duplicates are absent."""),
        code("""audit = json.loads((RUN / 'pairing_audit.json').read_text(encoding='utf-8'))
display(pd.DataFrame([{'samples': audit['samples'], **audit['split_counts'],
                       'exact_duplicate_groups': len(audit['exact_caption_duplicates']),
                       'normalized_duplicate_groups': len(audit['normalized_caption_duplicates']),
                       'repeated_video_groups': len(audit['repeated_videos'])}]))
display(Markdown((RUN / 'analysis/cases.md').read_text(encoding='utf-8')))"""),
        md("""## Scope and interpretation

Most true pairs remain outside the top ten despite improvement over both baselines.
Fixed classification-trained encoders, coarse graph features, available-audio
selection and the lack of artist-disjoint evaluation limit interpretation.
DistilBERT previously trained on 3,864 captions, including 1,089 additional
training-only captions without available paired audio; GraphSAGE trained on
2,775 paired clips. Held-out IDs retain their original splits.

No subjective quality or completed listening assessment is claimed. The original
proposal updates encoders; this projection-only experiment is a documented scope
deviation. See `report/task4_scope_deviation.md` and the extension results."""),
        md("""## Zero-shot comparisons and ten-query listening package

CLAP is a fixed pretrained audio/text comparator, not a graph encoder. The
caption-tag comparison separately uses positive/negative text prompt similarity,
with no local training or threshold tuning. The five-listener study is prepared
but has no real responses yet. Follow `report/task4_listening_guide.md`."""),
        code("""clap_root = ROOT / 'results/task4/clap_zeroshot'
clap = json.loads((clap_root / 'metrics.json').read_text(encoding='utf-8'))
assert recompute(clap_root / 'embeddings.npz') == clap['metrics']
display(pd.DataFrame({k: clap['metrics'][k] for k in ['text_to_graph', 'graph_to_text']}).T.rename(index={'text_to_graph':'text_to_audio_CLAP','graph_to_text':'audio_to_text_CLAP'}))
display(pd.read_csv(ROOT / 'results/task4/zeroshot_tags/comparison.csv'))
study = ROOT / 'results/task4/listening_study'
protocol = json.loads((study / 'protocol.json').read_text(encoding='utf-8'))
print('Human evaluation:', protocol['status'], '| Required listeners:', protocol['required_listeners'])
print('Each listener rates', protocol['trials_per_listener'], 'clips across', protocol['queries'], 'queries.')
display(Markdown((study / 'ten_caption_queries.md').read_text(encoding='utf-8')))"""),
        md("""## Optional synthetic execution check

Disabled by default. Random synthetic features validate execution only and do
not provide music-retrieval evidence. This cell uses a fresh temporary output
and evaluates validation only."""),
        code("""RUN_SYNTHETIC_DEMO = False
if RUN_SYNTHETIC_DEMO:
    import tempfile
    from src.task3.data import make_demo
    from src.task4.train import train
    output = Path(tempfile.mkdtemp(prefix='task4_demo_'))
    features = make_demo(output / 'features.pt')
    print(train(features, output / 'run', epochs=2, device='cpu'))"""),
    ]
    notebook.metadata = dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"))
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    notebook = build()
    if args.execute:
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
    nbf.write(notebook, root / "notebooks/task4_retrieval.ipynb")
    print(f"Wrote Task 4 notebook; executed={args.execute}")
