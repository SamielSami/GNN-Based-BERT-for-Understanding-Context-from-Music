from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("""# Task 3 — measured frozen GNN–BERT fusion

Review the completed five-head, three-seed comparison on 3,964 real MusicCaps
clips and 30 independent AudioSet labels. Default execution reads saved results;
it does not retrain or tune against test data. See `report/task3_results.md` for
interpretation and `docs/task3/README.md` for complete reproduction commands."""),
    code("""from pathlib import Path
import sys, json, re
import pandas as pd
from IPython.display import Image, display, Markdown
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / 'results/task3/available_run1'
assert (RUN / 'experiment.json').exists(), 'Run the real-data suite described in docs/task3/README.md first.'
experiment = json.loads((RUN / 'experiment.json').read_text(encoding='utf-8'))
selection = json.loads((RUN / 'selection.json').read_text(encoding='utf-8'))
verification = json.loads((RUN / 'verification.json').read_text(encoding='utf-8'))
print('Real data:', not experiment['synthetic'])
print('Seeds:', selection['seeds'])
print('Final evaluation complete:', experiment['evaluated_test'])
display(pd.DataFrame([{k: v for k, v in verification.items() if k not in {'checks', 'feature_sha256', 'selection_sha256'}}]))"""),
    md("""## Controlled comparison

Every head uses identical paired samples, labels and frozen encoder outputs.
DistilBERT was previously fine-tuned on 3,864 original training captions;
GraphSAGE on the 2,775 available-audio training clips. Held-out video IDs remain
disjoint from both training partitions. Only fusion-head initialization and batch
order change across seeds 42, 43 and 44.

Architecture selection uses mean validation Macro-F1. All test scores below
come from predeclared frozen controls with validation-selected thresholds.
Standard deviations describe head-seed variability, not confidence intervals."""),
    code("""comparison = pd.read_csv(RUN / 'comparison.csv')
aggregate = pd.read_csv(RUN / 'comparison_aggregate.csv')
display(aggregate[['mode', 'validation_macro_f1_mean', 'test_macro_f1_mean', 'test_macro_f1_std', 'test_micro_f1_mean', 'test_map_mean']])
display(comparison[['mode', 'seed', 'best_epoch', 'threshold', 'validation_macro_f1', 'test_macro_f1', 'test_micro_f1', 'test_map']])
display(Image(filename=str(RUN / 'analysis/ablation.png')))
display(Markdown('**Frozen validation selection**'))
print(json.dumps({k: v for k, v in selection.items() if k not in {'runs'}}, indent=2))"""),
    md("""## Measured learning curves and embeddings

Curves come from saved epoch logs. t-SNE shows the validation-selected fusion and
each seed's validation winner; axes are descriptive and do not measure class
separation quality or semantic understanding."""),
    code("""display(Image(filename=str(RUN / 'analysis/fusion_embeddings.png')))
for seed in selection['seeds']:
    display(Markdown(f'### Seed {seed}'))
    display(Image(filename=str(RUN / f'seed{seed}/validation_curves.png')))
    display(Image(filename=str(RUN / f'seed{seed}/training_curves.png')))
    display(Image(filename=str(RUN / f'seed{seed}/embeddings.png')))"""),
    md("""## Three evidence-grounded cases

The first three validation IDs are fixed in cache order. These are diagnostics
of the seed-42 cross-attention head, irrespective of the winning architecture.
Token and graph-node vector interventions measure model sensitivity. They do
not establish causal importance of words, sounds or instruments; contextual
information remains in other tokens, and graph edges remain unchanged."""),
    code("""case_root = RUN / 'case_studies'
for path in sorted(case_root.glob('*.md')):
    display(Markdown(re.sub(r'!\\[[^\\]]*\\]\\([^)]*\\)', '', path.read_text(encoding='utf-8'))))
for path in sorted(case_root.glob('*.png')):
    display(Image(filename=str(path)))"""),
    md("""## Interpretation and limitations

Read the measured report for fusion gains or negative findings, failure patterns
and exact timings. Independent targets remove mechanical caption-derived label
leakage, but captions naturally mention labeled instruments and sound types.
Missing-audio selection bias, incomplete annotations, modest graph features and
reused encoders limit generalization. Attention and t-SNE are descriptive."""),
    code("""report_text = (ROOT / 'report/task3_results.md').read_text(encoding='utf-8')
display(Markdown(re.sub(r'!\\[[^\\]]*\\]\\([^)]*\\)', '', report_text)))"""),
    md("""## Fusion inference from a saved checkpoint

Restore the validation-selected fusion head for the predeclared first seed and
run the first three validation clips through it. These inputs are the aligned,
frozen DistilBERT tokens and GraphSAGE representations, prepared from real audio
and captions. This demonstrates cached-pair inference; raw new audio/captions
must first use the same tokenizer, graph preprocessing and frozen encoders.
The saved threshold and label order are used unchanged."""),
    code("""import torch
from src.task3.data import load_features, feature_fingerprint
from src.task3.models import FusionClassifier
from src.task3.evaluate import predict_features
torch.set_num_threads(6)
entry = next(r for r in selection['runs']
             if r['mode'] == selection['selected_fusion'] and r['seed'] == selection['seeds'][0])
checkpoint_path = RUN / entry['checkpoint']
assert feature_fingerprint(checkpoint_path) == entry['checkpoint_sha256']
saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
features_path = Path(experiment['features'])
if not features_path.exists():
    features_path = ROOT / 'data/processed/task3_available/features.pt'
assert feature_fingerprint(features_path) == saved['feature_sha256']
data = load_features(features_path, mmap=True)
assert data['label_names'] == saved['label_names']
model = FusionClassifier(**saved['dimensions'], mode=saved['mode']).eval()
model.load_state_dict(saved['state_dict'])
indices = [i for i, split in enumerate(data['splits']) if split == 'validation'][:3]
probabilities, embeddings = predict_features(model, data, indices, batch_size=3, device='cpu')
names = json.loads((ROOT / 'src/task1/audioset_names.json').read_text(encoding='utf-8'))
rows = []
for i, scores in zip(indices, probabilities):
    rows.append({'sample_id': data['sample_ids'][i], 'caption': data['texts'][i],
                 'true_labels': [names.get(tag, tag) for tag, y in zip(saved['label_names'], data['labels'][i]) if y],
                 'predictions': {names.get(tag, tag): round(float(score), 4)
                                 for tag, score in zip(saved['label_names'], scores) if score >= saved['threshold']}})
print('Head:', saved['mode'], '| Threshold:', saved['threshold'], '| Embeddings:', embeddings.shape)
display(pd.DataFrame(rows))
del data"""),
    md("""## Optional offline demonstration

Set `RUN_SYNTHETIC_DEMO` to `True` to exercise training in a fresh temporary
directory. Random synthetic features are pipeline diagnostics, never real-data
results. Full reproduction commands live in `docs/task3/README.md`."""),
    code("""RUN_SYNTHETIC_DEMO = False
if RUN_SYNTHETIC_DEMO:
    from datetime import datetime
    from src.task3.data import make_demo
    from src.task3.train import run_experiments
    output = ROOT / 'tmp' / ('task3_demo_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    features = make_demo(output.with_suffix('.features.pt'))
    demo = run_experiments(features, output, epochs=2, device='cpu', evaluate_test=False)
    display(pd.read_csv(output / 'comparison.csv'))
    print('Synthetic:', demo['provenance']['synthetic'])"""),
]
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}}
nbf.write(nb, Path('notebooks/task3_fusion.ipynb'))
