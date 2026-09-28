"""Build the real audio/caption to joint-fusion inference demonstration."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
nb.cells = [
    md("""# Music context: audio and caption to prediction

This demonstration starts from a real local MusicCaps audio clip and its caption,
recomputes segment features and graph edges, and forwards both modalities through
the validation-selected joint fusion checkpoint. It uses the first validation
clip in processed-manifest order, fixed independently of predictions. No training,
threshold tuning, downloads or test-set inference is performed.

Run `src.task3.pipeline train`, then `evaluate`, using `configs/task3.yaml` before
opening this notebook. Dependencies are the completed `results/task3/joint_run2`,
the manifests and training normalization referenced by its run configuration,
and the first validation clip's local WAV file. See `docs/task3/README.md`."""),
    code("""from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
import torch
from IPython.display import display, Markdown, Audio
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.task3.joint import verify_joint, restore_model
from src.task2.dataset import resolve_record_path
RUN = ROOT / 'results/task3/joint_run2'
torch.set_num_threads(6)
verification = verify_joint(RUN)
assert verification['evaluated_test']
config = json.loads((RUN / 'run_config.json').read_text())
selection = json.loads((RUN / 'selection.json').read_text())
manifest_path = Path(config['processed_manifest'])
manifest = json.loads(manifest_path.read_text())
record = next(r for r in manifest['records'] if r['split'] == 'validation')
audio_path = resolve_record_path(manifest_path, record['audio_path'])
assert audio_path.is_file(), f'Required local audio is missing: {audio_path}'
display(Markdown(f\"**Clip:** `{record['sample_id']}`\\n\\n**Caption:** {record['text']}\"))
print('Audio:', audio_path)
print('Local interval:', record['audio_offset_seconds'], 'seconds; duration:', record['audio_duration_seconds'])"""),
    md("""## Rebuild the audio graph

Load the declared local interval, convert to mono at 22.05 kHz and peak-normalize.
Extract one-second segment features (log-mel statistics, chroma, MFCC/deltas and
three spectral/energy descriptors). Apply the saved training-only mean and
standard deviation, then build the checkpoint's graph variant. Normalization is
never refitted on this clip. The original video timestamps differ from the local
offset because the supplied WAV file is already trimmed."""),
    code("""from src.task2.features import AudioFeatureConfig, load_audio, extract_track_features_from_waveform
from src.task2.graphs import build_segment_graph, validate_graph
feature_config = AudioFeatureConfig(**manifest['feature_config'])
waveform = load_audio(audio_path, feature_config.sample_rate,
                      offset_seconds=record['audio_offset_seconds'],
                      duration_seconds=record['audio_duration_seconds'])
raw_features = extract_track_features_from_waveform(waveform, feature_config)
normalization_path = resolve_record_path(manifest_path, manifest['normalization_path'])
normalization = json.loads(normalization_path.read_text())
assert normalization['fitted_on'] == 'training nodes only'
mean = np.asarray(normalization['mean'], dtype=np.float32)
std = np.asarray(normalization['std'], dtype=np.float32)
node_features = (raw_features - mean) / std
sample_seed = manifest['seed'] + int(hashlib.sha1(record['sample_id'].encode()).hexdigest()[:8], 16)
graph = build_segment_graph(node_features, variant=config['graph_variant'],
                            top_k=manifest['top_k'], seed=sample_seed)
display(validate_graph(graph.x, graph.edge_index))
display(Audio(data=waveform, rate=feature_config.sample_rate))
# The graph used for inference above was rebuilt from audio. The cache is only an audit reference.
reference_graph = torch.load(resolve_record_path(manifest_path, record['processed_path']), weights_only=True, map_location='cpu')
torch.testing.assert_close(graph.x, reference_graph['x'], atol=1e-5, rtol=1e-5)
torch.testing.assert_close(graph.edge_index, reference_graph['edge_indices'][config['graph_variant']], atol=0, rtol=0)
print('Reconstructed graph matches the cached training-pipeline representation.')"""),
    md("""## Encode the caption and graph, fuse, predict

Restore the full selected fusion model and its tokenizer. The caption goes through
DistilBERT, the reconstructed graph goes through GraphSAGE, and projected outputs
go through the trained fusion head. Apply the saved validation threshold and
label order. Annotations are used only to display prediction outcomes, never as
inputs to the model."""),
    code("""from transformers import AutoTokenizer
from src.task3.cases import label_outcomes
mode = selection['selected_fusion']
saved = torch.load(RUN / mode / 'best_model.pt', weights_only=True, map_location='cpu', mmap=True)
model = restore_model(saved, 'cpu')
tokenizer = AutoTokenizer.from_pretrained(RUN / 'tokenizer', local_files_only=True)
tokens = tokenizer([record['text']], padding='max_length', truncation=True,
                   max_length=config['max_length'], return_tensors='pt')
with torch.inference_mode():
    probabilities = model(graph, tokens).sigmoid()[0].numpy()
names = json.loads((ROOT / 'src/task1/audioset_names.json').read_text())
outcomes = label_outcomes(probabilities, record['labels'], saved['label_names'], names, saved['threshold'])
display(pd.DataFrame(outcomes).sort_values('probability', ascending=False))
print('Selected fusion:', mode, '| Saved threshold:', saved['threshold'])
print('Predicted labels:', ', '.join(r['name'] for r in outcomes if r['predicted']))
with np.load(RUN / mode / 'validation_predictions.npz') as expected:
    assert str(expected['sample_ids'][0]) == record['sample_id']
    np.testing.assert_allclose(probabilities, expected['probabilities'][0], atol=1e-5, rtol=1e-5)
print('Raw-audio/caption inference matches the saved prediction for this validation ID.')"""),
    md("""## Interpretation

This is one real validation example, not a new benchmark estimate. See the joint
report for held-out Macro-F1, Micro-F1, mAP and PR-AUC and the inference notebook
`task3_joint.ipynb` for three fixed cases. A predicted label without an AudioSet
positive annotation is counted as a false positive, although annotations can be
incomplete. Neither attention weights nor an embedding visualization establishes
causal importance. MusicCaps substitution approval and submission packaging
remain separate requirements.

To use another local audio/caption pair, apply the same feature configuration,
saved normalization, tokenizer and selected checkpoint. Set the correct local
offset and duration; the equality checks against this validation sample's saved
artifacts apply only to the fixed demonstration clip."""),
]
destination = ROOT / "notebooks/demo_context.ipynb"
nbf.write(nb, destination)
print(destination)
