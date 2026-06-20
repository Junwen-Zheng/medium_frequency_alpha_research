# Push this reworked repo to GitHub

This zip includes a `.git` folder with a multi-commit history. To preserve that history on GitHub, extract it into a fresh folder and push from there.

```bash
cd medium_frequency_alpha_research

git log --oneline --decorate --max-count=10

git remote remove origin 2>/dev/null || true
git remote add origin https://github.com/Junwen-Zheng/medium_frequency_alpha_research.git

git push -u origin main --force-with-lease
```

`--force-with-lease` is needed only if the GitHub repo already has your previous single-commit version. It rewrites the remote branch to the cleaner multi-commit history in this folder.

If you do not want to rewrite history, copy the files into your existing repo and commit normally. That is safer, but the GitHub history will still show the original single drop plus new commits.
