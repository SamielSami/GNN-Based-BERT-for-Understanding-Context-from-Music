"""Task 1: MusicCaps audit, BERT proxy-tag training, evaluation, and inference.

The package is intentionally split into small executable modules:

- :mod:`src.task1.audit` validates MusicCaps viability.
- :mod:`src.task1.data` builds the caption-to-tag proxy CSV.
- :mod:`src.task1.train` trains and evaluates the BERT classifier.
- :mod:`src.task1.predict` loads a checkpoint for text inference.

See ``docs/task1/README.md`` for the architecture, commands, and limitations.
"""

