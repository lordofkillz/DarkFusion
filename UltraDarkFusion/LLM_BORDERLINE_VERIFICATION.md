# Borderline label review with Ollama

Dataset Analysis can ask a local vision model to review visually unusual
annotations. This is supporting evidence, not proof that a gameplay label is
wrong. Inspect the full frame and class before removing any annotation.

DarkFusion checks the local Ollama service at `http://localhost:11434`.
Its setup helper can install/start Ollama and download a supported model if
one is unavailable. First-use setup may take time and use additional disk space
and GPU memory; subsequent checks reuse the service, model, and cached results.

For manual setup with the supported fallback model:

```powershell
ollama pull llava:7b
ollama serve
```

Watch the application log for `LLM borderline verifier ready`. If setup or
verification is unavailable, DarkFusion continues without that evidence.
Do not treat HUD text, icons, scenery, or an unlabeled detection as an automatic
missing annotation. SAM3 shape analysis is a separate check using the existing
SAM3 checkpoint.
