import tempfile
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf

from src.task4.experiment import read, write
from src.task4.listening import prepare, summarize, ten_queries
from src.task4.zeroshot_tags import tag_scores


class ExtensionTests(unittest.TestCase):
    def test_ten_queries_and_five_complete_blinded_responses(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            audio=root/'fixture.wav'
            sf.write(audio,np.zeros(4800),48000)
            source=dict(records=[dict(sample_id=f'id{i}',text=f'Caption {i}',audio_path=str(audio),
                                      audio_offset_seconds=0,audio_duration_seconds=0.1) for i in range(10)])
            manifest=root/'manifest.json';write(manifest,source)
            embeddings=root/'embeddings.npz'
            np.savez(embeddings,text=np.eye(10),graph=np.eye(10),sample_ids=[f'id{i}' for i in range(10)])
            output=root/'study'
            protocol=prepare(manifest,embeddings,embeddings,output)
            self.assertEqual(protocol['queries'],10)
            # Identical methods share query/clip judgments instead of duplicated listening.
            self.assertEqual(protocol['trials_per_listener'],30)
            query=read(output/'ten_caption_queries.json')['examples'][0]
            self.assertEqual(query['methods']['graph']['top3'][0]['sample_id'],'id0')
            self.assertEqual(query['methods']['graph']['paired_rank'],1)
            page=(output/'participants/L01.html').read_text(encoding='utf-8')
            self.assertNotIn('"roles"',page)
            self.assertNotIn('"sample_id"',page)
            ratings=root/'ratings';ratings.mkdir()
            with self.assertRaisesRegex(ValueError,'five distinct'):
                summarize(output,ratings)
            key=read(output/'coordinator_key.json')
            for listener in protocol['listener_ids']:
                write(ratings/f'{listener}_ratings.json',dict(study_id=protocol['study_id'],listener_id=listener,consent=True,
                      ratings=[dict(trial_id=t['trial_id'],rating=3,listened=True) for t in key['trials']]))
            result=summarize(output,ratings)
            self.assertEqual(result['methods']['graph']['mean'],3)
            self.assertEqual(result['methods']['clap']['sample_std'],0)
            bad=read(ratings/'L01_ratings.json');bad['ratings'][0]['rating']=None
            write(ratings/'L01_ratings.json',bad)
            with self.assertRaisesRegex(ValueError,'1–5 rating'):
                summarize(output,ratings)

    def test_misaligned_comparator_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,ids in [('a',list(range(10))),('b',list(reversed(range(10))))]:
                np.savez(root/f'{name}.npz',text=np.eye(10),graph=np.eye(10),sample_ids=ids)
            with self.assertRaisesRegex(ValueError,'galleries differ'):
                ten_queries(dict(records=[]),dict(a=root/'a.npz',b=root/'b.npz'))

    def test_caption_tag_margin_has_fixed_decision_boundary(self):
        scores=tag_scores(np.eye(2),np.eye(2),-np.eye(2))
        self.assertGreater(scores[0,0],0.5)
        self.assertEqual(scores[0,1],0.5)
        self.assertTrue(np.isfinite(scores).all())
