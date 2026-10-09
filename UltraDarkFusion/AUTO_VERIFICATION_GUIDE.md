# Reviewed-image tracking

In the **Nested Label Review** viewer, Previous/Next navigation records the
current image as reviewed before moving on. Refresh alone does not record it.
Ordinary image navigation elsewhere in DarkFusion does not use this hook.

The record is stored in your dataset at `.darkfusion/verified_images.json`.
Dataset Analysis's visual-outlier scan skips images recorded there. This is
review bookkeeping, not proof that annotations are correct: inspect and fix an
image before advancing. It does not create missing-object annotations.

Each dataset has its own list. Back it up with the dataset. If you want to
revisit a reviewed image, remove its record; clear the list only when you want
all previously reviewed images included in future visual-outlier scans again.

For a source installation, you can manage records from the application folder:

```python
from darkfusion_verified_images import VerifiedImagesTracker

tracker = VerifiedImagesTracker()
dataset = r"D:\Datasets\Example"
tracker.get_verified_count(dataset)
tracker.remove_from_verified(dataset, "images/001.jpg")
# Deliberately reset all review bookkeeping, without changing label files:
# tracker.clear_verified_list(dataset)
```

If a previously reviewed image still appears, check that the scan uses the same
dataset root and that the metadata file is readable. Other analysis/review
tools are not necessarily filtered by this list.
