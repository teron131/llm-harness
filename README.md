```bash
uv add "llm_harness@git+https://github.com/teron131/llm-harness"
```

```bash
git remote add llm-harness https://github.com/teron131/llm-harness.git
git fetch llm-harness
git subtree add --prefix llm-harness llm-harness main --squash
uv add --editable ./vendor/llm-harness
```
