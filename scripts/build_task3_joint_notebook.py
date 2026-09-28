"""Build the real joint-fusion review and live-encoder inference notebook."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
nb.cells = [
    md("""# Task 3: joint GNN-BERT fusion

Review the five-mode joint experiment and run three real validation captions and
normalized audio graphs through its saved encoders and fusion head. This notebook
does not retrain, tune thresholds, or repeat test inference. Run the commands in
`docs/task3/README.md` first. The raw-audio preprocessing stage is upstream of this
demonstration; the input graph cache and source captions must be available."""),
    code("""from pathlib import Path
import json, re, sys
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / 'results/task3/joint_run2'
from src.task3.joint import verify_joint
verification = verify_joint(RUN)
assert verification['evaluated_test'], 'Finish the predeclared suite evaluation first.'
config = json.loads((RUN / 'run_config.json').read_text())
selection = json.loads((RUN / 'selection.json').read_text())
display(verification)
print('Overall selection:', selection['selected_mode'])
print('Fusion selection:', selection['selected_fusion'])"""),
    md("""## Matched ablation comparison

DistilBERT-only, GraphSAGE-only, concatenation, gating and graph-query-to-text-token
cross-attention share paired IDs, labels and splits. Attention masks exclude text
padding. The final transformer block, GraphSAGE and fusion/head update jointly
where present; lower transformer weights remain frozen. Every mode starts from
the original Task 1/2 encoders and its corresponding frozen seed-42 head. This
is continued fine-tuning with one seed; it is separate from the earlier three-seed
frozen-head comparison. Validation alone selects epochs, thresholds and modes."""),
    code("""columns = ['mode', 'best_epoch', 'threshold', 'validation_macro_f1', 'test_macro_f1',
           'test_micro_f1', 'test_mean_average_precision', 'test_mean_pr_auc_trapezoidal']
display(pd.read_csv(RUN / 'comparison.csv')[columns])
display(Image(filename=str(RUN / 'analysis/ablation.png')))
for mode in config['modes']:
    display(Markdown(f'### {mode}'))
    display(Image(filename=str(RUN / mode / 'training_curves.png')))
    proof = json.loads((RUN / mode / 'training_proof.json').read_text())
    display(pd.DataFrame(proof['selected_checkpoint_changes']).T)
    assert proof['frozen_text_sha256_before'] == proof['frozen_text_sha256_after']"""),
    md("""## Genre and mood embeddings

The same validation t-SNE coordinates are colored using true positive AudioSet
annotations. Overlapping genres are explicit and also appear in individual
panels. Missing annotations are not confirmed negatives. Only Angry music is
available as a selected mood label, so broad mood discrimination is not shown.
Neither t-SNE nor attention weights establish causal explanations."""),
    code("""display(Image(filename=str(RUN / 'semantic_plots/genre_mood_tsne.png')))
display(Image(filename=str(RUN / 'semantic_plots/genre_label_panels.png')))
semantic = json.loads((RUN / 'semantic_plots/metadata.json').read_text())
display({group: semantic[group] for group in ['genre', 'mood']})"""),
    md("""## Three fixed validation cases and failure analysis

Cases are the first three validation IDs in processed-manifest order, selected
independently of outcomes. The analysis contrasts the selected fusion with both
unimodal controls. Validation cases are diagnostic; held-out aggregate scores
estimate generalization. Incomplete labels can produce apparent false positives."""),
    code("""display(Markdown((RUN / 'analysis/cases.md').read_text(encoding='utf-8')))
analysis_text = (RUN / 'analysis/README.md').read_text(encoding='utf-8')
display(Markdown(re.sub(r'!\\[[^\\]]*\\]\\([^)]*\\)', '', analysis_text)))"""),
    md("""## Live encoder inference from the joint checkpoint

Restore the validation-selected fusion model, tokenize the real captions, and
forward normalized audio node features and edges through GraphSAGE and caption
tokens through DistilBERT. No cached encoder embeddings are used here. The saved
threshold and ordered label vocabulary are retained. These three validation
predictions are checked against the saved prediction artifact."""),
    code("""import torch
from torch.utils.data import Subset
from transformers import AutoTokenizer
from src.task3.joint import restore_model, PairedDataset, predict
torch.set_num_threads(6)
mode = selection['selected_fusion']
saved = torch.load(RUN / mode / 'best_model.pt', map_location='cpu', weights_only=True, mmap=True)
model = restore_model(saved, 'cpu')
tokenizer = AutoTokenizer.from_pretrained(RUN / 'tokenizer', local_files_only=True)
source = json.loads(Path(config['source_manifest']).read_text(encoding='utf-8'))
validation = PairedDataset(config['processed_manifest'], source, 'validation',
                           config['graph_variant'], tokenizer, config['max_length'])
prediction = predict(model, Subset(validation, [0, 1, 2]), 3, 'cpu')
with np.load(RUN / mode / 'validation_predictions.npz') as expected:
    np.testing.assert_array_equal(prediction['sample_ids'], expected['sample_ids'][:3])
    np.testing.assert_array_equal(prediction['targets'], expected['targets'][:3])
    np.testing.assert_allclose(prediction['probabilities'], expected['probabilities'][:3], atol=1e-5, rtol=1e-5)
names = json.loads((ROOT / 'src/task1/audioset_names.json').read_text(encoding='utf-8'))
rows = []
for i, identifier in enumerate(prediction['sample_ids']):
    record = validation.records[i]
    predicted = [names.get(label, label) for label, p in zip(saved['label_names'], prediction['probabilities'][i]) if p >= saved['threshold']]
    annotated = [names.get(label, label) for label, y in zip(saved['label_names'], prediction['targets'][i]) if y]
    rows.append(dict(sample_id=identifier, caption=record['text'], predicted=', '.join(predicted), annotated=', '.join(annotated)))
display(pd.DataFrame(rows))
print('Three live encoder predictions match saved validation predictions; threshold:', saved['threshold'])"""),
    md("""## Scope

MusicCaps substitutes for the assignment's named datasets; instructor acceptance
remains undocumented. The text encoder's earlier training included 1,089 extra
training-only captions. Graphs compress the audio into ten segments. Missing
audio and incomplete labels affect this cohort, and artist-disjoint generalization
has not been established. No statistical significance is claimed from one joint
seed. See the submission audit for final report and packaging requirements."""),
]
destination = ROOT / "notebooks/task3_joint.ipynb"
nbf.write(nb, destination)
print(destination)
