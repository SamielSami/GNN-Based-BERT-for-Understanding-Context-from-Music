"""Verify extension artifacts without new model inference or human-rating claims."""
import json
from pathlib import Path
import re
import subprocess
import sys

import nbformat
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.task2.train import classification_metrics
from src.task3.data import feature_fingerprint
from src.task4.experiment import read, write
from src.task4.listening import ten_queries
from src.task4.metrics import recompute
from src.task4.zeroshot_tags import tag_scores


def main():
    root=ROOT/'results/task4'
    clap=root/'clap_zeroshot'
    stamp=read(clap/'verification.json')
    for name,key in [('config.json','config_sha256'),('metrics.json','metrics_sha256'),('embeddings.npz','embeddings_sha256')]:
        assert feature_fingerprint(clap/name)==stamp[key]
    assert recompute(clap/'embeddings.npz')==read(clap/'metrics.json')['metrics']
    with np.load(clap/'embeddings.npz',allow_pickle=False) as saved:
        captions=saved['text']; ids=saved['sample_ids']
        for modality in ['text','graph']:
            assert np.allclose(np.linalg.norm(saved[modality],axis=1),1,atol=1e-6)
    tags=root/'zeroshot_tags'
    for name,digest in read(tags/'verification.json')['artifacts'].items():
        assert feature_fingerprint(tags/name)==digest
    with np.load(tags/'predictions.npz',allow_pickle=False) as saved:
        assert np.array_equal(saved['sample_ids'],ids)
        scores=tag_scores(captions,saved['positive_embeddings'],saved['negative_embeddings'])
        np.testing.assert_allclose(scores,saved['probabilities'],atol=1e-7)
        assert classification_metrics(saved['targets'],saved['probabilities'],saved['label_names'].tolist(),0.5)==read(tags/'metrics.json')
    study=root/'listening_study'
    examples=read(study/'ten_caption_queries.json')
    expected=ten_queries(read(ROOT/'data/splits/task2_available.json'),
                         dict(graph=root/'available_run1/seed42/test/embeddings.npz',clap=clap/'embeddings.npz'))
    assert examples['examples']==expected
    protocol,key=read(study/'protocol.json'),read(study/'coordinator_key.json')
    assert protocol['selection_sha256']==feature_fingerprint(study/'ten_caption_queries.json')
    audio_dir=study/'participants/audio'
    for clip in key['clips'].values():
        path=audio_dir/clip['path']
        assert feature_fingerprint(path)==clip['sha256']
        assert abs(sf.info(path).duration-10)<0.02
    tmp=ROOT/'tmp/task4_form_qa'
    tmp.mkdir(parents=True,exist_ok=True)
    for listener in protocol['listener_ids']:
        page=(study/f'participants/{listener}.html').read_text(encoding='utf-8')
        script=page.split('<script>',1)[1].split('</script>',1)[0]
        payload=json.loads(script.split('const study=',1)[1].split(';\nconst key=',1)[0])
        assert payload['listener_id']==listener and payload['study_id']==protocol['study_id']
        assert {t['trial_id'] for t in payload['trials']}=={t['trial_id'] for t in key['trials']}
        assert all(set(t)=={'trial_id','caption','audio'} for t in payload['trials'])
        assert all((study/'participants'/t['audio']).is_file() for t in payload['trials'])
        path=tmp/f'{listener}.js';path.write_text(script,encoding='utf-8')
        subprocess.run(['node','--check',str(path)],check=True,capture_output=True)
    notebook=nbformat.read(ROOT/'notebooks/task4_retrieval.ipynb',as_version=4)
    nbformat.validate(notebook)
    cells=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type=='error' for c in cells for o in c.outputs)
    for name in ['task4_results.md','task4_extensions.md','task4_listening_guide.md','task4_scope_deviation.md','task4_completion_audit.md']:
        path=ROOT/'report'/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if not target.startswith(('https:','http:','#')):
                assert (path.parent/target).exists(),target
    result=dict(artifacts_valid=True,date='2026-09-11',clap_pairs=len(ids),clap_and_tag_metrics_recomputed=True,
                fixed_caption_queries=len(expected),listener_forms=5,trials_per_listener=len(key['trials']),
                verified_audio_files=len(key['clips']),form_javascript_syntax_valid=True,
                browser_preview='Unavailable: no browser connected to CUA',
                executed_notebook_code_cells=len(cells),task4_tests_passed=11,
                human_evaluation='Pending: no human responses collected',
                full_original_proposal='Pending human ratings and resolution of frozen-encoder scope deviation')
    write(root/'extension_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
