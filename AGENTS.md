## Model support and upstream porting

### Scope
- This repository is a customized `llama.cpp` fork with significant local changes in:
  - TurboQuant-related `ggml/` code
  - TriAttention / KV-cache logic
  - server/runtime behavior tied to the current fork layout
- Treat this repo as a **custom runtime**, not as a clean mirror of upstream `llama.cpp`.

### Rule for new model support
- When a newly released model fails to load because the architecture is unknown, missing, partially supported, or behaves incorrectly, **do not default to a full upstream merge**.
- Preferred strategy is:
  1. identify the exact upstream commits that introduced support for that model family,
  2. determine the exact files changed,
  3. port only the minimum verified logic into this repo,
  4. rebuild and validate after each phase.

### Why
- Large upstream merges or rebases are high risk in this repo because they can overwrite or break TurboQuant, TriAttention, or other local optimizations.
- Small verified backports are preferred over broad syncs unless a full rebase is explicitly requested.

### Porting workflow for any new architecture
- Phase 1: architecture recognition and model loading.
- Phase 2: tensor parsing / runtime graph fixes.
- Phase 3: tokenizer / vocab / Unicode fixes.
- Phase 4: chat template, parser, and tool-calling fixes.
- Phase 5: assistant / MTP / speculative decoding support only if needed.

### Validation workflow
- After each phase:
  - rebuild `llama-server`,
  - test the target model again,
  - stop and verify before moving to the next phase.
- Do not batch unrelated fixes into one large edit.
- Prefer minimal, reversible changes.

### Build guidance
- Known working Windows CUDA configure command:
  - `cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86`
- Here, `86` is the NVIDIA CUDA compute capability target for RTX 30-series GPUs, not the Windows `x64` CPU architecture.
- A successful build typically ends with output similar to:
  - `llama-server.vcxproj -> ...\build\bin\Release\llama-server.exe`

### Performance interpretation
- Do not use tok/s alone as proof that a feature such as TriAttention is enabled or disabled.
- Do not assume model-load errors are caused by VRAM or launch flags before checking architecture support and tokenizer/runtime compatibility.

### Ask first
- Ask before:
  - rebasing onto upstream `llama.cpp`,
  - replacing major source files wholesale,
  - modifying `ggml/` TurboQuant internals,
  - modifying TriAttention / KV-cache internals,
  - changing global build flags or default runtime behavior.

### Change style
- Make the smallest change that unblocks the target model.
- Preserve local optimizations unless the task explicitly requires changing them.
- When support for a new model is added, update this file only with durable lessons that apply to future model ports too.

For new model families, prefer verified surgical backports over full upstream merges.