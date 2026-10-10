# Model and runtime safety

Install DarkFusion from this repository or its official GitHub releases. The
Windows installer checks downloaded payload and model-bundle SHA-256 hashes
before extraction. Do not substitute files from an unknown source.

Only open models and configuration files you trust. PyTorch `.pt`/`.pth`,
TorchScript models, and Python model configs can contain executable content;
DarkFusion is not a sandbox for untrusted models. A checksum proves that a file
matches the expected download, not that an arbitrary model is safe.

DarkFusion's Hugging Face loaders disable remote custom code. Its model-loading
paths also validate Accelerate checkpoint indexes before loading: shard paths
must stay within the checkpoint folder (or the same official Hugging Face model
cache), and shards must be regular files. This is an application-side mitigation
for [CVE-2026-69112](https://github.com/advisories/GHSA-4j2p-28q2-5m79), not a claim
that Accelerate's upstream implementation is fixed. Do not modify checkpoint
files or cache links while a model is loading.

Use the pinned requirements in a fresh, dedicated Python 3.12 environment and
keep the interpreter and operating system security-patched. Never upgrade an
environment while it is running training. Windows ONNX inference uses DirectML;
the separately bundled PyTorch CUDA 13 runtime requires NVIDIA driver branch
580 or newer for CUDA training. Only one ONNX Runtime distribution may be
installed in an environment.

To report a suspected vulnerability, use the repository's private GitHub
security-reporting option if available; otherwise contact the maintainer without
posting exploit code, credentials, or private datasets in a public issue.
