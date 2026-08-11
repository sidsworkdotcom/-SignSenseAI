# Development notes

## Versioning policy

As per the guidance of our faculty, Prof. Nitish Kumar, this repository is
updated incrementally as the project progresses rather than uploaded in one
go. All intermediate versions are pushed — trained model checkpoints, code
revisions, and experiment outputs — so the commit history reflects the actual
development timeline. The repository may look cluttered with superseded files
during development; once the project reaches its final version, older
intermediate versions will be pruned and only the final version retained.

The latest state of the project is always the tip of `main`. Work-in-progress
details, training numbers, and debugging notes for each session are kept in
`docs/LOGS.md`.

## Deployment plan

The browser demo (`web/`) will be hosted at:

**https://signsenseai.siddheshgupta.com**

Setup: static hosting (GitHub Pages) with a CNAME record on the
siddheshgupta.com DNS pointing the subdomain at the Pages site. HTTPS is
required for browser camera access and is provided by the host. The demo is
fully client-side — the exported model (`web/model.json`) runs in the
visitor's browser, so no backend, GPU, or inference server is needed and no
video leaves the user's device.

Deployment happens after the word-mode vocabulary is finalized and the LSTM
is exported for in-browser inference.
