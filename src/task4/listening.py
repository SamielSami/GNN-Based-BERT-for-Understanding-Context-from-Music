"""Ten fixed caption queries and a blinded five-listener 1–5 relevance study."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from src.task2.features import load_audio
from src.task3.data import feature_fingerprint
from .experiment import read, write
from .metrics import normalized_pairs, paired_ranks
from .train import empty_output


def ten_queries(features_metadata, embeddings):
    """First ten gallery IDs: deterministic and independent of retrieval success."""
    records = {r["sample_id"]: r for r in features_metadata["records"]}
    methods, ids = {}, None
    for name, path in embeddings.items():
        with np.load(path, allow_pickle=False) as saved:
            current = saved["sample_ids"].tolist()
            if ids is not None and current != ids:
                raise ValueError("Compared retrieval galleries differ")
            ids = current
            methods[name] = normalized_pairs(saved["text"], saved["graph"])
    if ids is None or len(ids) < 10:
        raise ValueError("Ten distinct caption queries required")
    result = []
    ranks = {name: paired_ranks(*pair) for name, pair in methods.items()}
    for i in range(10):
        query = dict(query_id=ids[i], caption=records[ids[i]]["text"], methods={})
        for name, (text, gallery) in methods.items():
            scores = text[i] @ gallery.T
            order = np.lexsort((np.arange(len(ids)) == i, -scores))[:3]
            query["methods"][name] = dict(paired_rank=int(ranks[name][i]), top3=[
                dict(rank=rank, sample_id=ids[j], caption=records[ids[j]]["text"],
                     cosine=float(scores[j]), paired_positive=bool(j == i)) for rank, j in enumerate(order, 1)])
        result.append(query)
    return result


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Caption and music listening study</title><style>
body{font:17px system-ui,sans-serif;background:#f3f5f7;color:#182b37;max-width:850px;margin:35px auto;padding:0 20px}
article,header{background:white;padding:24px;margin:18px 0;border-radius:12px;border:1px solid #d8e0e6}
audio{width:100%;margin:12px 0}button{padding:12px 20px;background:#125c75;color:white;border:0;border-radius:6px;cursor:pointer}
select{font:inherit;padding:7px}label{display:block;margin:12px 0}#status{font-weight:600}p{line-height:1.55}
</style><header><h1>Does the music match the caption?</h1>
<p>Listen to every clip using headphones at a comfortable volume. Rate the overall match to the caption:
1 = no match; 2 = weak; 3 = partial; 4 = good; 5 = very strong. Consider instruments, vocals, rhythm and mood.
You are judging relevance, not recording quality. Some captions describe details you may not hear.</p>
<p>Participation is voluntary. You may stop without submitting. Do not enter your name. Audio stays on this device;
the downloaded response contains your assigned listener code and ratings. Do not inspect other listeners’ forms
or discuss ratings until everyone finishes. Replay any clip as needed. Expect about 15–20 minutes.</p>
<label><input type="checkbox" id="consent"> I consent to these anonymous ratings being used in the course report.</label>
<p id="status"></p></header><main id="trials"></main><button id="export">Download completed ratings</button>
<p id="message" role="status"></p><script>
const study=STUDY_DATA;
const key='music-study-'+study.study_id+'-'+study.listener_id;
let answers={};try{answers=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
function persist(){localStorage.setItem(key,JSON.stringify(answers));document.getElementById('status').textContent=
study.listener_id+' — '+study.trials.filter(t=>answers[t.trial_id]?.rating && answers[t.trial_id]?.listened).length+' / '+study.trials.length+' clips rated';}
for(const [i,t] of study.trials.entries()){
 const a=document.createElement('article');const h=document.createElement('h2');h.textContent='Clip '+(i+1);
 const p=document.createElement('p');p.textContent=t.caption;
 const audio=document.createElement('audio');audio.controls=true;audio.preload='none';audio.src=t.audio;
 const label=document.createElement('label');label.textContent='Match rating: ';
 const select=document.createElement('select');select.setAttribute('aria-label','Match rating for clip '+(i+1));
 for(let j=0;j<=5;j++){const o=document.createElement('option');o.value=j||'';o.textContent=j||'Choose';select.append(o)}
 select.value=answers[t.trial_id]?.rating||'';
 const listened=document.createElement('label');const cb=document.createElement('input');cb.type='checkbox';cb.checked=!!answers[t.trial_id]?.listened;
 listened.append(cb,document.createTextNode(' I listened to this clip.'));
 const update=()=>{answers[t.trial_id]={rating:Number(select.value)||null,listened:cb.checked};persist()};select.onchange=update;cb.onchange=update;
 label.append(select);a.append(h,p,audio,label,listened);document.getElementById('trials').append(a);
}persist();
document.getElementById('export').onclick=()=>{
 const message=document.getElementById('message');
 if(!document.getElementById('consent').checked){message.textContent='Consent is required to submit.';return}
 if(study.trials.some(t=>!answers[t.trial_id]?.listened||!answers[t.trial_id]?.rating)){message.textContent='Listen to and rate every clip before submitting.';return}
 const result={study_id:study.study_id,listener_id:study.listener_id,consent:true,submitted_at:new Date().toISOString(),
 ratings:study.trials.map(t=>({trial_id:t.trial_id,...answers[t.trial_id]}))};
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
 a.download=study.listener_id+'_ratings.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);
 message.textContent='Ratings downloaded. Return that JSON file to the study coordinator.';
};</script></html>'''


def prepare(manifest, graph_embeddings, clap_embeddings, output_dir):
    source = read(manifest)
    methods = dict(graph=graph_embeddings, clap=clap_embeddings)
    queries = ten_queries(source, methods)
    root = empty_output(output_dir)
    write(root / "ten_caption_queries.json", dict(selection="First ten test IDs in unchanged cache order; seed42 graph model",
                                                  examples=queries, embedding_hashes={k:feature_fingerprint(p) for k,p in methods.items()}))
    lines = ["# Ten caption queries → top-three clips", "", "Fixed first ten test queries; graph seed 42 and pinned zero-shot CLAP. "
             "Selected by ID order, not success. Descriptions are dataset annotations, not human assessments.", ""]
    for q in queries:
        lines += [f"## `{q['query_id']}`", "", q["caption"], ""]
        for method, result in q["methods"].items():
            lines += [f"**{method}: paired rank {result['paired_rank']}.**", ""]
            for match in result["top3"]:
                lines.append(f"- {match['rank']}. `{match['sample_id']}` — cosine {match['cosine']:.4f}; "
                             f"{'paired' if match['paired_positive'] else 'nonpaired'}. {match['caption']}")
            lines.append("")
    (root / "ten_caption_queries.md").write_text("\n".join(lines), encoding="utf-8")
    public = root / "participants"
    audio_dir = public / "audio"
    audio_dir.mkdir(parents=True)
    records = {r["sample_id"]:r for r in source["records"]}
    trials, clips = [], {}
    for q in queries:
        matches = {}
        for method, result in q["methods"].items():
            for match in result["top3"]:
                matches.setdefault(match["sample_id"], []).append(dict(method=method, rank=match["rank"]))
        for sid, roles in matches.items():
            token = hashlib.sha256((q["query_id"] + ':' + sid).encode()).hexdigest()[:16]
            clip = hashlib.sha256(sid.encode()).hexdigest()[:16] + ".wav"
            trials.append(dict(trial_id=token, query_id=q["query_id"], caption=q["caption"], sample_id=sid,
                               audio="audio/" + clip, roles=roles))
            if sid not in clips:
                record = records[sid]
                wave = load_audio(record["audio_path"], 48000, offset_seconds=record["audio_offset_seconds"],
                                  duration_seconds=record["audio_duration_seconds"])
                sf.write(audio_dir / clip, wave, 48000, subtype="PCM_16")
                clips[sid] = dict(path=clip, sha256=feature_fingerprint(audio_dir / clip))
    study_id = feature_fingerprint(root / "ten_caption_queries.json")[:16]
    protocol = dict(study_id=study_id, status="awaiting_human_ratings", required_listeners=5,
                    listener_ids=[f"L{i:02d}" for i in range(1,6)], queries=10, trials_per_listener=len(trials),
                    scale={"1":"no match", "2":"weak", "3":"partial", "4":"good", "5":"very strong"},
                    aggregation="Mean of top-three ratings per query per method, then equal query mean per listener; mean/sample SD across listeners",
                    sampling="First ten test IDs, no quality selection; graph seed42, fixed CLAP",
                    blinding="Opaque clip IDs, no method/rank/candidate caption; deterministic shuffled order separately per listener",
                    shared_candidates="Same query/clip pair rated once and reused for both methods if applicable",
                    selection_sha256=feature_fingerprint(root / "ten_caption_queries.json"))
    write(root / "protocol.json", protocol)
    write(root / "coordinator_key.json", dict(study_id=study_id, trials=trials, clips=clips))
    for i, listener in enumerate(protocol["listener_ids"]):
        order = list(trials)
        random.Random(9100+i).shuffle(order)
        payload = dict(study_id=study_id, listener_id=listener,
                       trials=[{k:t[k] for k in ("trial_id","caption","audio")} for t in order])
        encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
        (public / f"{listener}.html").write_text(PAGE.replace("STUDY_DATA",encoded), encoding="utf-8")
    (root / "README.md").write_text(
        "# Five-listener evaluation — awaiting real ratings\n\n"
        "Give each of five different people their assigned L01–L05 HTML file and the accompanying audio folder. "
        "Open the file in a browser, consent, listen, rate every clip 1–5, then download the ratings JSON. "
        "Keep coordinator_key.json and the qualitative examples hidden until everyone finishes. "
        "Distribute only locally permitted audio; this package has not been published or sent to anyone.\n\n"
        "Store the five returned files in a ratings folder and run:\n\n"
        "```powershell\npython -m src.task4.listening summarize --output-dir results/task4/listening_study "
        "--ratings-dir PATH_TO_RETURNED_JSON_FILES\n```\n\n"
        "Use five distinct humans, one assigned code each; do not let one person complete multiple codes. "
        "The coordinator must confirm this externally—files alone cannot prove human identity. "
        "No real ratings exist yet. Empty/missing/incomplete responses are not results.\n", encoding="utf-8")
    return protocol


def summarize(output_dir, ratings_dir):
    root = Path(output_dir)
    protocol, key = read(root / "protocol.json"), read(root / "coordinator_key.json")
    if (key["study_id"] != protocol["study_id"] or
            feature_fingerprint(root / "ten_caption_queries.json") != protocol["selection_sha256"]):
        raise ValueError("Listening study selection changed")
    expected = {t["trial_id"]:t for t in key["trials"]}
    responses = [read(p) for p in sorted(Path(ratings_dir).glob("*_ratings.json"))]
    listeners = [r.get("listener_id") for r in responses]
    if len(listeners) != 5 or set(listeners) != set(protocol["listener_ids"]):
        raise ValueError("Exactly five distinct assigned listener responses are required")
    summaries = []
    for response in responses:
        if response.get("study_id") != protocol["study_id"] or response.get("consent") is not True:
            raise ValueError("Study identity and consent must match")
        ratings = response.get("ratings", [])
        if len(ratings) != len(expected) or {r.get("trial_id") for r in ratings} != set(expected):
            raise ValueError("Missing, duplicate or unknown trials")
        method_scores = {}
        for rating in ratings:
            if type(rating.get("rating")) is not int or not 1 <= rating["rating"] <= 5 or rating.get("listened") is not True:
                raise ValueError("Every trial requires a real 1–5 rating and listened confirmation")
            trial = expected[rating["trial_id"]]
            for role in trial["roles"]:
                method_scores.setdefault(role["method"], {}).setdefault(trial["query_id"], []).append(rating["rating"])
        for method, queries in method_scores.items():
            if len(queries) != 10 or any(len(scores) != 3 for scores in queries.values()):
                raise ValueError("Each method must contain ten queries with three ranked clips")
            summaries.append(dict(listener_id=response["listener_id"], method=method,
                                  mean_rating=float(np.mean([np.mean(s) for s in queries.values()]))))
    methods = {m: [r["mean_rating"] for r in summaries if r["method"]==m] for m in ("graph","clap")}
    result = dict(status="five_responses_complete_identity_requires_coordinator_confirmation", listeners=5, per_listener=summaries,
                  methods={m:dict(mean=float(np.mean(v)), sample_std=float(np.std(v,ddof=1))) for m,v in methods.items()},
                  response_hashes={p.name:feature_fingerprint(p) for p in sorted(Path(ratings_dir).glob("*_ratings.json"))})
    write(root / "human_results.json", result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","summarize"])
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--manifest",default="data/splits/task2_available.json")
    parser.add_argument("--graph-embeddings",default="results/task4/available_run1/seed42/test/embeddings.npz")
    parser.add_argument("--clap-embeddings",default="results/task4/clap_zeroshot/embeddings.npz")
    parser.add_argument("--ratings-dir")
    args=vars(parser.parse_args()); command=args.pop("command"); ratings=args.pop("ratings_dir")
    print(prepare(**args) if command=="prepare" else summarize(args["output_dir"],ratings))
