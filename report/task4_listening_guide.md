# How to run the five-listener evaluation

The automatic retrieval metric asks, “Did the model return the exact paired
clip?” The listening study asks, “Does this clip sound like the caption?” A
different clip may still be a good semantic match. The proposal requires at
least five human listeners, each rating matches from 1 to 5.

## What you need to do

1. Recruit **five different people**, such as classmates or friends. They do
   not need machine-learning knowledge. Ask them to use headphones if available.
2. Open `results/task4/listening_study/participants/`. Assign one person each
   to **L01.html, L02.html, L03.html, L04.html and L05.html**. Give each person
   their assigned HTML file and the accompanying `audio` folder, keeping them
   together. They can also take turns on your computer using different forms.
3. Each listener opens their file in a browser, reads the instructions, and
   checks the consent box. They read each caption, listen to the clip, choose
   a rating, and check “I listened to this clip.”
4. Listeners work independently. Do not tell them which model produced a
   result, show candidate captions, or suggest the “correct” rating. One person
   should not fill in multiple listener codes.
5. After all clips are rated, they click **Download completed ratings**. Collect
   the five files named `L01_ratings.json` through `L05_ratings.json`.
6. Put those files in one folder and run the command below, or give their local
   folder path to the assistant for analysis.

```powershell
python -m src.task4.listening summarize --output-dir results/task4/listening_study --ratings-dir C:/path/to/returned_ratings
```

## Rating scale

| Rating | Meaning |
|---|---|
| 1 | No match |
| 2 | Weak match |
| 3 | Partial match |
| 4 | Good match |
| 5 | Very strong match |

Consider instruments, vocals, rhythm and mood. Judge caption relevance rather
than recording quality or whether you personally like the music. Listen to
the whole short clip before rating. Replay if needed; there is no timed test.

## What the package contains

There are **ten fixed caption queries**, with the graph model's top three clips
and CLAP's top three clips for each. The study has up to **60 short clips per
listener**, fewer when both models return the same clip for the same query.
Allow roughly **15–20 minutes**. Order is shuffled separately for each listener.
The forms conceal method names, retrieval ranks, source IDs and candidate captions.

Keep `coordinator_key.json`, `ten_caption_queries.json` and the qualitative
example report hidden from listeners until they finish. Share only their HTML
form and locally permitted study audio. The package has not been published or
sent to anyone automatically.

## How results are calculated

For each listener and model, the script averages three clip ratings per query,
then averages across the ten queries. It reports the mean and sample standard
deviation of the **five listener scores**. A query/clip pair returned by both
models is rated once and contributes to both. It does not treat all individual
ratings as independent listeners or claim statistical significance.

The script checks five distinct assigned codes, matching study IDs, consent,
complete trial lists and valid integer ratings. You must confirm that the codes
belong to **five actual people**; files cannot prove that by themselves.

**Current status: no human ratings collected.** The forms and protocol are
preparation, not completed evaluation results. Do not put invented averages or
software-test ratings into the course report.
