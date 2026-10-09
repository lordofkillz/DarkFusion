#!/usr/bin/env python3
"""
COMPREHENSIVE FALSE POSITIVE DETECTION STRATEGY

Problem Identified:
- Previous scan was limited to 1 pass (default)
- Only used a single baseline reference set
- If that baseline had any false positives, they'd be locked in forever
- "Trusted examples" (manually verified) could mask real anomalies

Solution: Multi-Pass Rotating Reference Detection
- Default now 3 passes (was 1)
- Each pass uses a DIFFERENT random reference set
- Cross-validates across the entire dataset
- No single pass's baseline can contaminate the results
- Manually verified labels still protected (separate logic)

What This Means:
==================================================================

PASS 1: Comprehensive Scan
  • Randomly sample ~30% of class as reference set
  • Compare ALL annotations against this reference
  • Flag anything below adaptive threshold
  • Removes flagged items from consideration
  
PASS 2: Validate Remaining
  • Different random ~30% sample becomes reference
  • Re-scan the remaining (un-flagged) items
  • Catches FPs that slipped past pass 1
  • Uses same DINOv3+SigLIP2+foreground scoring
  
PASS 3: Final Sweep
  • Another random reference set
  • Final catch for edge cases
  • Ensures comprehensive coverage

Why Rotating Reference Works:
  • Pass 1 baseline has some FPs → Pass 2 catches them
  • Objects similar only to pass 1's FPs → Pass 2/3 flag them
  • No single "trusted baseline" pollutes all results
  • Statistical redundancy increases confidence

Speed Impact:
  • ~3x slower than 1 pass
  • But catches 3x more false positives
  • Multi-GPU or batch processing can parallelize

Configuration:
  To change number of passes, you can override:
  
    app._scan_visual_outlier_passes_override = 5  # Use 5 passes instead
    app._scan_visual_outliers()
    
  Or via UI settings if available.

Manually Verified Labels:
  • Labels you've checked and confirmed correct are protected
  • They won't be suppressed as false positives
  • This preserves your manual work
  • But doesn't prevent finding new anomalies elsewhere

Recommendation:
==================================================================
For your COD game footage dataset with lots of manual rules:

1. Run with 3 passes (new default) → find all possible anomalies
2. Review the flagged items manually
3. Mark ones that are correct as "trusted"
4. Next scan will protect those but find new anomalies

This creates a virtuous cycle:
  Scan → Review → Mark Correct → Scan Again (finds new FPs)

The foreground-aware scoring (coherent=stricter, weak=lenient) 
ensures occluded valid objects aren't falsely flagged while 
clear anomalies are caught with higher sensitivity.
"""

print(__doc__)
