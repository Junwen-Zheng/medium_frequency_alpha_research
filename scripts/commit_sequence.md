# Suggested commit sequence

James pointed out that a single commit makes the repo look like it appeared in one drop rather than through an actual research process.

This zip already contains a local `.git` history with multiple commits. If you push from the extracted folder, GitHub should show that history.

If you instead copy files into an existing repo, do not commit everything in one batch. Use a sequence like this:

```bash
git add README.md .gitignore
git commit -m "Reframe project as research case study"

git add src/workflow.py src/reporting.py
git commit -m "Add validation-first research pipeline"

git add docs/research_log docs/experiments reports/research_report.md
git commit -m "Document research process and failed experiments"

git add docs/github_review_checklist.md docs/research_note_template.md scripts/commit_sequence.md
git commit -m "Add reviewer checklist and research notes"

git add tests
git commit -m "Add validation and feature pipeline tests"
```

Do not fake old history. From now on, make real commits as you add experiments, logs, and results.
