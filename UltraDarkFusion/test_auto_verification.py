#!/usr/bin/env python3
"""
Test auto-verification system for label maker navigation.
"""

import os
import tempfile
import json
from darkfusion_verified_images import VerifiedImagesTracker, get_verified_images_for_scan


def test_verified_images_tracker():
    """Test basic tracker functionality."""
    print("\n" + "="*70)
    print("TEST: Verified Images Tracker")
    print("="*70)
    
    # Create temp dataset directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = VerifiedImagesTracker()
        
        # Test 1: Mark an image as verified
        print("\n1. Mark image as verified...")
        image1 = "images/001.jpg"
        was_new = tracker.mark_image_verified(tmpdir, image1)
        assert was_new, "Image should be marked as new"
        print(f"   ✓ Image marked: {image1}")
        
        # Test 2: Check it's verified
        print("\n2. Check image is verified...")
        is_verified = tracker.is_verified(tmpdir, image1)
        assert is_verified, "Image should be verified"
        print(f"   ✓ Image verified: {is_verified}")
        
        # Test 3: Mark same image again (should not be new)
        print("\n3. Mark same image again...")
        was_new = tracker.mark_image_verified(tmpdir, image1)
        assert not was_new, "Image should not be new on second marking"
        print(f"   ✓ Already marked (not new): {not was_new}")
        
        # Test 4: Mark multiple images
        print("\n4. Mark multiple images...")
        images = ["images/002.jpg", "images/003.jpg", "images/004.jpg"]
        for img in images:
            tracker.mark_image_verified(tmpdir, img)
        count = tracker.get_verified_count(tmpdir)
        assert count == 4, f"Should have 4 verified images, got {count}"
        print(f"   ✓ Marked {len(images)} more images, total verified: {count}")
        
        # Test 5: Get all verified images
        print("\n5. Get all verified images...")
        verified_list = tracker.get_verified_images(tmpdir)
        assert len(verified_list) == 4, f"Should have 4 verified, got {len(verified_list)}"
        print(f"   ✓ Retrieved {len(verified_list)} verified images:")
        for v in verified_list:
            print(f"     - {v}")
        
        # Test 6: Get verified images for scan
        print("\n6. Test get_verified_images_for_scan()...")
        verified_set = get_verified_images_for_scan(tmpdir)
        assert len(verified_set) == 4, f"Should have 4 in set, got {len(verified_set)}"
        print(f"   ✓ Retrieved {len(verified_set)} images as set for filtering")
        
        # Test 7: Remove from verified
        print("\n7. Remove image from verified...")
        was_removed = tracker.remove_from_verified(tmpdir, image1)
        assert was_removed, "Image should be removed"
        count_after = tracker.get_verified_count(tmpdir)
        assert count_after == 3, f"Should have 3 after removal, got {count_after}"
        print(f"   ✓ Image removed, count now: {count_after}")
        
        # Test 8: Path normalization (Windows vs Unix)
        print("\n8. Test path normalization...")
        # Test with backslashes
        image_with_backslash = "images\\005.jpg"
        tracker.mark_image_verified(tmpdir, image_with_backslash)
        
        # Check with forward slashes
        image_with_forward = "images/005.jpg"
        is_verified_forward = tracker.is_verified(tmpdir, image_with_forward)
        assert is_verified_forward, "Should find image regardless of path separator"
        print(f"   ✓ Path normalization works (backslash → forward slash)")
        
        # Test 9: Verify metadata file was created
        print("\n9. Verify metadata file creation...")
        metadata_file = tracker.get_verified_file_path(tmpdir)
        assert os.path.exists(metadata_file), "Metadata file should exist"
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        assert "verified_images" in data, "Should have verified_images key"
        assert len(data["verified_images"]) > 0, "Should have some verified images"
        print(f"   ✓ Metadata file created: {metadata_file}")
        print(f"   ✓ File contains {len(data['verified_images'])} verified images")
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)


if __name__ == "__main__":
    test_verified_images_tracker()
