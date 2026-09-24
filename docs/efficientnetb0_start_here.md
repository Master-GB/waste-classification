# EfficientNetB0: start here

This starter extends `Master-GB/waste-classification` at commit
`fc0649a30753ea7d4dac13f3973250f536486491` (main, inspected 2026-09-23).
It adds an EfficientNetB0 frozen-backbone baseline alongside the ResNet50 work.
It is not a trained model and contains no claimed waste-classification results.

## What the team already provides

- Ten classes in `configs/base.yaml`.
- Frozen CSV manifests: 13,833 training, 2,964 validation, 2,965 test images.
- Shared RGB conversion, training augmentation, 224-pixel inputs and ImageNet normalization.
- A shared PyTorch trainer that selects the best checkpoint by validation loss.
- Classification metrics and prediction collection.

Use the manifests exactly as committed. Do not run `02_create_splits.ipynb`
again, split your own copy independently, or train on another member's subset.
All four models use the full shared training set.

## Before training: obtain the exact shared images

Raw images are intentionally excluded from Git. Ask the teammate who prepared
the data for the exact dataset archive or shared folder corresponding to these
manifests, and the source version/cleaning notes. The earlier Kaggle version-8
link alone does not establish which files the teammate used.

Place the extracted category folders directly inside:

`data/raw/garbage-dataset/`

For example, `data/raw/garbage-dataset/battery/battery_1.jpg` must exist.
Avoid accidentally nesting another `garbage-dataset` folder inside it.
Alternatively, edit `DATASET_ROOT` in each notebook or set `WASTE_DATASET_ROOT`.
File names and category folders must match the CSVs. Path checks do not verify
that image contents are identical; obtaining the same archive matters.

The checked-in audit reports zero unreadable images and zero byte-identical
duplicate groups, but 1,669 images sharing 814 repeated perceptual hashes.
These are similarity candidates, not proof that every pair is a duplicate.
The split notebook performs a stratified file-level split and does not show
grouping/removal of these candidates. Ask whether the shared cleaned copy
resolved them. If a split correction is needed, the team must update it once
for everyone and rerun affected experiments; do not change just your split.

## 1. Clone and create your branch on your Mac

Open Terminal. These commands assume you have not cloned this repository yet:

```bash
mkdir -p ~/Projects
cd ~/Projects
git clone https://github.com/Master-GB/waste-classification.git
cd waste-classification
git switch -c feature/model-efficientnetb0
git branch --show-current
```

If already cloned, finish or stash your own uncommitted work first, then use:

```bash
cd /path/to/your/waste-classification
git status
git switch main
git pull --ff-only origin main
git switch -c feature/model-efficientnetb0
```

If that branch already exists locally, use `git switch feature/model-efficientnetb0`
instead of creating it again. Do not force-push or overwrite a teammate's branch.

Set your actual identity **for this repository**, replacing the placeholders:

```bash
git config user.name "Your Full Name"
git config user.email "YOUR_VERIFIED_GITHUB_EMAIL_OR_GITHUB_NOREPLY_EMAIL"
git config --get user.name
git config --get user.email
```

Use the email from your own GitHub Settings > Emails so GitHub can associate
your commits with you. This does not grant repository write access.

Push your new branch:

```bash
git push -u origin feature/model-efficientnetb0
```

The inspected ChatGPT GitHub connection is `radixian` and reports `push: false`.
No remote branch or commit was created through that connection. If your Mac
account also lacks write access, ask Master-GB to add your intended GitHub
account as a collaborator and accept the invitation. Authenticate through
GitHub's supported sign-in/credential flow; do not put a token in notebooks,
source files, command URLs, or chat. A GitHub account password is not an HTTPS
Git credential.

## 2. Add this starter without replacing existing files

The ZIP contains only new files under a `waste-classification/` folder.
Copy its contents into your clone, merging folders. Do not replace an entire
existing `src`, `notebooks` or `configs` folder in Finder. A terminal copy is:

```bash
cp -R ~/Downloads/EfficientNetB0_Starter/waste-classification/. ~/Projects/waste-classification/
cd ~/Projects/waste-classification
git status --short
```

Adjust the source path if your browser extracted the ZIP elsewhere. Existing
ResNet50 code and shared CSVs should not appear as modified files.

## 3. Create a portable Python environment

Use a supported Python version, for example Python 3.12, installed on your Mac.
Check `python3 --version` first. The team's current `requirements.txt` is a
Windows/CUDA environment snapshot: it includes `pywinpty` and `+cu132` wheels.
Do not use that file as your Mac installation recipe or overwrite it.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements-efficientnet.txt
python -m ipykernel install --user --name waste-classification --display-name "Waste Classification"
```

Install torch and torchvision together as a matching pair. If the installer
cannot find a wheel for your Mac/Python combination, consult the official
PyTorch platform selector rather than attempting to install Windows/CUDA wheels.
The minimal requirements file is not a lock file; the training notebook records
the actual environment for each run.

Apple GPU availability check:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('Apple GPU available:', torch.backends.mps.is_available())"
```

The notebooks choose CUDA if present, otherwise Apple MPS if available, otherwise
CPU. CPU execution works but can make full training slow. No CUDA installation
is needed on your Mac. If using a cloud GPU, use the same repository branch,
manifests and exact images; preserve the run folder and checkpoint before the
session ends. Do not assume a cloud runtime's disk is permanent.

## 4. Review the model and make your implementation commits

`src/models/efficientnetb0.py` loads ImageNet EfficientNetB0, replaces its final
classifier with ten outputs, and freezes the feature extractor. Its wrapper
keeps feature-extractor BatchNorm statistics fixed when the shared trainer calls
`model.train()`. Classifier dropout stays active during training.

The current ResNet50 factory freezes parameter gradients but not BatchNorm
running statistics. Discuss aligning or documenting this difference in the
group comparison. No teammate code was changed by this starter.

The starter keeps the team's shared augmentation and bilinear evaluation resize.
The torchvision B0 weight preset uses bicubic evaluation resizing; the common
preprocessing choice is recorded in the notebook so the comparison is explicit.

After installing and reviewing the setup files:

```bash
git add requirements-efficientnet.txt docs/efficientnetb0_start_here.md
git diff --cached
git commit -m "chore(efficientnet): add portable setup and implementation guide"
git push
```

After reviewing the architecture and configuration:

```bash
git add src/models/efficientnetb0.py configs/efficientnetb0.yaml
git diff --cached
git commit -m "feat(efficientnet): add pretrained B0 classifier and frozen baseline config"
git push
```

The config starts with the same baseline settings as the existing ResNet50 run:
10 epochs, batch size 32, AdamW, learning rate 0.001, weight decay 0.0001, seed 42.
These are starting settings for comparison, not guaranteed optimal values.
If you lower batch size for memory limits, record that change.

## 5. Run the checks and record genuine progress

Run the offline implementation test from the repository root:

```bash
python -m unittest discover -s tests -p 'test_efficientnetb0.py' -v
```

This uses synthetic inputs and no pretrained download. It checks output shape,
gradient updates, unchanged frozen weights/BatchNorm state, checkpoint loading,
and unfreezing. It is not an accuracy evaluation.

Open VS Code using File > Open Folder and choose your clone. Install the
Microsoft Python and Jupyter extensions if needed, open
`notebooks/06_efficientnetb0_smoke_test.ipynb`, and select the **Waste
Classification** kernel. Alternatively run `jupyter lab` from your activated
terminal and open the notebook there.

Run all cells in order. The first pretrained model construction downloads its
weights if not cached. Expected shapes at batch size 32 are `(32, 3, 224, 224)`
for inputs and `(32, 10)` for output scores. Do not expect good predictions yet:
the ten-class head is newly initialized.

After it really passes, save the notebook and commit:

```bash
git add tests/test_efficientnetb0.py notebooks/06_efficientnetb0_smoke_test.ipynb experiments/efficientnetb0/smoke_check.json
git diff --cached
git commit -m "test(efficientnet): verify frozen backbone and shared-data forward pass"
git push
```

If it fails, fix the concrete issue and record that change honestly. Do not
commit a successful-check message or made-up output before running it.

## 6. Train the frozen baseline

Review and commit the training workflow before a long run:

```bash
git add notebooks/07_efficientnetb0_frozen_training.ipynb
git diff --cached
git commit -m "feat(efficientnet): add frozen training and validation workflow"
git push
```

Open that notebook in the same kernel and run it in order. The trainer saves
the best checkpoint by validation loss. After training, the notebook restores
that checkpoint and saves validation metrics, predictions, a classification
report, confusion matrix, learning curves, class mapping, settings, source
hashes and environment versions.

Each completed experiment lives in a new timestamped folder under
`results/efficientnetb0/`. Confirm `training_complete: true` in its metadata.
History is saved after the training cell completes. An interruption can leave
a best checkpoint but does not create a completed run or resume automatically.
Use a fresh run or implement/document proper resumption before continuing.

After a completed, inspected run:

```bash
git add notebooks/07_efficientnetb0_frozen_training.ipynb results/efficientnetb0/
git diff --cached --stat
git commit -m "experiment(efficientnet): record frozen baseline validation results"
git push
```

The existing ignore rules exclude `.pth`, `.pt`, the dataset and your `.venv`.
Back up `best.pth` in the team's agreed storage and record its link in your
experiment notes. Git pushes alone do not back up that model file.

## 7. Next meaningful milestones

Do these as actual subsequent work, not empty commits or invented dates:

| Completed change | Example commit message |
| --- | --- |
| Implement a fine-tuning experiment starting from the best frozen checkpoint | `feat(efficientnet): add fine-tuning experiment with lower learning rate` |
| Run fine-tuning, save and discuss observed validation results | `experiment(efficientnet): record fine-tuning validation results` |
| Compare completed runs using validation metrics and choose settings | `docs(efficientnet): explain baseline and fine-tuning comparison` |
| After the group finalizes settings, implement final test evaluation | `feat(efficientnet): add final held-out test evaluation` |
| Run that evaluation and save the actual test outputs | `experiment(efficientnet): record final held-out test results` |

Fine-tuning means reloading your best checkpoint, calling
`model.set_backbone_trainable(True)` (or deliberately implementing partial
unfreezing), creating a new optimizer with a smaller learning rate such as
`1e-5`, and saving a separate experiment. Do not compare the last epoch of one
model with the best checkpoint of another. Do not select hyperparameters using
the test set.

A meaningful commit captures a completed change or real experimental evidence.
You do not need to commit every epoch, and a larger commit count is not proof
of a better contribution. Record AI assistance and reused team/pretrained code
as required by the assignment; make sure you can explain your changes.

## 8. Integration

Before starting new work, with your current changes committed, incorporate
shared updates carefully:

```bash
git fetch origin
git merge origin/main
```

Resolve conflicts by checking the actual changes. If manifests changed, discuss
whether earlier experiments must be rerun. Open a pull request from
`feature/model-efficientnetb0` to `main` when your work is ready for team review.
Keep training in your feature branch; do not push directly to main.

## Validation of this starter

Verified during preparation with Python 3.12, PyTorch 2.14.0+cpu and
torchvision 0.29.0+cpu:

- Offline model test passed (classifier gradients/updates, fixed frozen state,
  checkpoint reload and unfreezing).
- Both notebooks passed nbformat validation and code-cell syntax checks.
- All Python cells in both notebooks executed in order against a separate,
  tiny synthetic image fixture, including ImageNet weight loading, two training
  epochs, best-checkpoint restoration, CSV/JSON exports and plot generation.
- The two generated plot layouts were visually inspected.
- Original tracked repository files and manifests were unchanged.

This environment could not start a normal Jupyter TCP kernel, so code-cell
integration was checked directly through Python. The delivered notebooks remain
unexecuted: no synthetic outputs or fixture data are included. Full notebook
execution on the real shared images, Mac MPS execution and model accuracy remain
to be checked on your machine. Never use software-test metrics as assignment
results.

## References

- Shared repository: https://github.com/Master-GB/waste-classification
- EfficientNetB0: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
- PyTorch installation: https://pytorch.org/get-started/locally/
- Apple MPS: https://docs.pytorch.org/docs/stable/notes/mps.html
- Git identity: https://docs.github.com/en/get-started/git-basics/setting-your-username-in-git
- GitHub authentication: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github
