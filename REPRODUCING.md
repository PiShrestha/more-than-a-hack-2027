# Reproducing the encara_messi submission

## What is fixed

The inference code, six CNN checkpoints, decoder, and original CSV are unchanged
from `602b6139b8fe82d0256515102444534bf94b32d7`. Their SHA-256 hashes are recorded
in [submission/manifest.json](submission/manifest.json). The final technical
document is `submission/encara_messi.md`.

The expected CSV contains 199,319 rows and has SHA-256:

```text
557ea3435993e72a83839000e98799be75d268f91a8e1fc0432686ab9a27c19c
```

Verification and packaging use only Python's standard library, Python 3.10 or
later. Model reproduction uses Python 3.14 and the exact versions in
`requirements-solution.txt`. [The recorded Linux environment](submission/environment-linux.txt)
also lists transitive packages from the research machine, including CUDA packages
that are not needed for CPU inference. It is an environment record, not a portable
installation command. The complete reconstruction was checked on Linux
with eight CPU cores and a 32 GiB memory allocation. No GPU is needed for
submission inference. Allow several GiB of working space for converted data
and fresh caches. Runtime depends on hardware and download speed. Reserve at least 30 minutes for
a fresh-cache run on a shared cluster. One 15-minute release check timed out;
the retry passed in 7 minutes 46 seconds on another node. The
[verification record](submission/verification.json) includes both attempts.

## Verify the committed files

```bash
python3 tools/submission.py verify
python3 -m unittest discover -s tests -v
```

This checks file integrity, frame order, labels, and row count. It does not
measure football accuracy. If a hash differs, inspect the change before using
the files. Do not update the manifest just to make a failing check pass.

## Prepare the two upload files

```bash
python3 tools/submission.py package --output-dir dist/encara_messi
```

Upload the resulting `encara_messi.csv` and `encara_messi.md` in their respective
form fields. Use the frozen `possession-model` branch URL for the model source.
The packaging command preserves different existing destination files and exits
with an error. Choose a new output directory to keep a separate package.

The files match the prepared Downloads copies. Submission through the form is
a separate team action; the tools do not send a response or claim receipt.

## Reproduce from permitted inputs

Start from a clean checkout of this `submission` branch. Install Python 3.14,
then create an isolated environment:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-solution.txt
```

Place these two competition files in a private directory under `tactical/`:

- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_atd.xml`

The videos, homography, broadcast tracking, and Smart Tagging are not required.
Keep all competition assets out of Git. See DATA.md for permitted access.
Download the external files and verify every required raw input:

```bash
export MTAH_DATA_DIR=/absolute/path/to/private/data
bash possession/fetch_idsse.sh
python tools/submission.py check-inputs --data-dir "$MTAH_DATA_DIR"
```

The [input manifest](submission/input-manifest.json) identifies all 16 required
raw files by size and SHA-256. A missing, incomplete, or different download
fails before training. It contains file hashes, not the licensed inputs.

Run the frozen pipeline through the guarded wrapper:

```bash
python tools/submission.py reproduce \
  --data-dir "$MTAH_DATA_DIR" \
  --output-dir dist/reproduced
```

The wrapper checks the release files, Python version, installed dependencies,
and input hashes. It refuses `POSS_*` and `TCN_*` environment overrides. It then
runs the original `run.sh` with a temporary source copy, fresh caches, and links
to the verified raw inputs. The original raw data and release CSV are not
modified. LightGBM is trained with eight threads; the six committed CNNs are
loaded for CPU inference.

The original official validator runs inside that pipeline. Packaging succeeds
only if the regenerated CSV also matches the expected SHA-256 exactly. A
failure returns a nonzero exit code. Temporary files are removed on normal completion or a handled error. A hard
scheduler kill can leave private temporary caches under the temporary directory.

Do not enable the optional research commands in `run.sh`. They can replace
model weights or decoder settings. Leave them on a research branch and give
their outputs separate names.
