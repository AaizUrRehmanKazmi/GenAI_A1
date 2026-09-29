# Work in small checkpoints
1. Run the Docker smoke check and inspect `/docs` and four placeholder workspaces. Read each Dockerfile and explain build versus runtime stages.
2. Choose framework and tracking provider with a short evidence-based decision record; resolve and freeze the training environment for your hardware.
3. Research and implement only data preparation next. Verify disjoint splits, runtime corruptions, pixel alignment and deterministic evaluation. Review image grids before training.
4. Complete Task 1 research record and one short baseline, then Optuna. Save validation evidence before choosing a final configuration.
5. Complete Task 2, then initialize Task 3 from its checkpoints; independently work through Task 4.
6. Export and verify ONNX, design in Stitch, implement API/UI and rerun Docker checks with real models.
7. Final evaluation only after choices are frozen; write analysis, limitations and AI-use notes as you go.

Git repository is initialized locally. Create an empty GitHub repository in your account when ready, then set its actual URL with `git remote add origin URL`, commit reviewed files and push. No remote or submission has been created by this scaffold.

Before your first commit, configure your own repository-local identity:
```sh
git config user.name "Your Name"
git config user.email "Your GitHub email or GitHub noreply address"
git add .
git commit -m "Add assignment skeleton and Docker infrastructure"
```
