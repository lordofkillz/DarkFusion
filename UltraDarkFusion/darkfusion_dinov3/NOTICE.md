# DINOv3 inference runtime

Copyright (c) Meta Platforms, Inc. and affiliates.

This package contains inference code from https://github.com/facebookresearch/dinov3
at commit `6876159a11b4df116f30f667f8c9888617df0751`, under the accompanying
[DINOv3 License](LICENSE.md). These files are subject to that license.

Local changes: the import namespace is `darkfusion_dinov3`; package initializers
include only the layers used by the vision transformer; `utils.py` includes only
three inference helpers; two global torch Dynamo configuration assignments were
removed from `layers/block.py` to avoid changing the host application's compiler
settings. The model math and checkpoint parameter names are unchanged.

The local loader constructs the published LVD1689M Base/Large configurations and
loads verified checkpoints from the application's Sam folder. The review worker
downloads a missing Base or Large checkpoint from a pinned DarkFusion GitHub
release. Those unmodified Meta weights are distributed under the accompanying
DINOv3 License, also included with the release assets. No dependency on the debug folder,
torch.hub, network access, or an installed `dinov3` package is required.
