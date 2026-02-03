### Reproducible install (don’t edit `llm_harness` here)

```bash
uv add "llm_harness @ git+https://github.com/teron131/llm-harness.git"
uv sync
```

### Editable install in the same repo (submodule)

```bash
git submodule add https://github.com/teron131/llm-harness.git llm-harness
uv add --editable ./llm-harness
uv sync
```

### Fresh clone `other_repo` with submodule

```bash
git clone --recurse-submodules <other_repo-url>
uv sync
```

### Already cloned `other_repo` with submodule

```bash
git submodule update --init
uv sync
```
