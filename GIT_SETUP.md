# Git Setup Guide

## What's Tracked

✅ **Tracked (committed to git):**
- Source code (`src/`, `scripts/`)
- Configuration files (`configs/`)
- Documentation (`*.md`)
- Requirements (`requirements.txt`)
- Directory structure (`.gitkeep` files)
- Main manifest (`artifacts/manifests/manifest_v1.csv`)

❌ **Ignored (not committed):**
- Large data files (`data/`, `*.npy`)
- Model checkpoints (`*.pth`, `*.pt`)
- Virtual environment (`.venv311/`)
- Logs and temporary files
- IDE-specific files
- Backup files

## Initial Setup

```bash
# Already done:
git init

# Add all files (respects .gitignore)
git add .

# Create initial commit
git commit -m "Initial commit: ASL recognition pipeline"

# Add remote repository (replace with your repo URL)
git remote add origin https://github.com/yourusername/asl-model.git

# Push to remote
git push -u origin main
```

## Common Git Commands

### Check status
```bash
git status
```

### Add changes
```bash
# Add specific files
git add src/data/dataloader.py

# Add all changes
git add .
```

### Commit changes
```bash
git commit -m "Add baseline model training script"
```

### Push to remote
```bash
git push
```

### Pull from remote
```bash
git pull
```

### View history
```bash
git log --oneline
```

### Create a branch
```bash
git checkout -b feature/new-model
```

### Switch branches
```bash
git checkout main
```

## Recommended Workflow

1. **Before starting work:**
   ```bash
   git pull
   ```

2. **Make changes to code**

3. **Check what changed:**
   ```bash
   git status
   git diff
   ```

4. **Stage and commit:**
   ```bash
   git add .
   git commit -m "Descriptive message about changes"
   ```

5. **Push to remote:**
   ```bash
   git push
   ```

## Important Notes

- **Large files are ignored** - Data and models won't be committed
- **Manifest is tracked** - The CSV manifest is small enough to track
- **Directory structure preserved** - Empty folders kept with `.gitkeep`
- **Virtual env ignored** - Others will create their own with `requirements.txt`

## Sharing Your Project

When someone clones your repo, they'll need to:

```bash
# Clone the repo
git clone https://github.com/yourusername/asl-model.git
cd asl-model

# Create virtual environment
python3.11 -m venv .venv311
source .venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download/prepare data (they'll need to do this themselves)
# See README.md for data preparation steps
```

## .gitignore Highlights

- ✅ Ignores `data/` (too large for git)
- ✅ Ignores `artifacts/landmarks/*.npy` and `artifacts/features/*.npy`
- ✅ Ignores model checkpoints (`*.pth`, `*.pt`)
- ✅ Keeps manifest CSV (small, important metadata)
- ✅ Preserves folder structure with `.gitkeep` files
- ✅ Ignores virtual environment
- ✅ Ignores IDE files (`.vscode/`, `.idea/`)

## Best Practices

1. **Commit often** - Small, focused commits are better
2. **Write clear messages** - Describe what and why
3. **Don't commit large files** - Use `.gitignore`
4. **Pull before push** - Avoid conflicts
5. **Use branches** - For experimental features
6. **Review before commit** - Use `git diff` to check changes

## Example Commit Messages

Good:
- `Add LSTM baseline model`
- `Fix normalization bug in preprocessing`
- `Update dataloader to support variable-length sequences`
- `Add training script with early stopping`

Bad:
- `Update`
- `Fix bug`
- `Changes`
- `WIP`
