"""Pinned local checkpoint metadata; importing this file needs no ML runtime."""

SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
MODEL_CONFIGS = {
    "dinov3_base": {
        "label": "DINOv3 Base",
        "filename": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "sha256": "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c",
        "size": 342860279,
        "embedding_size": 768,
        "depth": 12,
        "heads": 12,
    },
    "dinov3_large": {
        "label": "DINOv3 Large",
        "filename": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "sha256": "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035",
        "size": 1213050671,
        "embedding_size": 1024,
        "depth": 24,
        "heads": 16,
    },
}
