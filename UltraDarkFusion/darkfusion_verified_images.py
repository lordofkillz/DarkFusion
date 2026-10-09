#!/usr/bin/env python3
"""
Verified Images Tracker

Manages automatic marking of images as verified when user navigates away
after cleaning them in Label Maker. Prevents re-flagging already-cleaned images
in future Dataset Analysis scans.
"""

import json
import os
from pathlib import Path
from datetime import datetime


class VerifiedImagesTracker:
    """Track images that have been verified/cleaned by user."""
    
    VERIFIED_METADATA_DIR = ".darkfusion"
    VERIFIED_IMAGES_FILE = "verified_images.json"
    
    def __init__(self, dataset_root=None):
        """Initialize tracker for given dataset root.
        
        Args:
            dataset_root: Path to dataset directory
        """
        self.dataset_root = dataset_root
        self.verified_data = {}
        if dataset_root:
            self.load_verified_list(dataset_root)
    
    def get_verified_file_path(self, dataset_root):
        """Get path to verified images JSON file."""
        metadata_dir = os.path.join(dataset_root, self.VERIFIED_METADATA_DIR)
        return os.path.join(metadata_dir, self.VERIFIED_IMAGES_FILE)
    
    def load_verified_list(self, dataset_root):
        """Load verified images list from dataset."""
        verified_file = self.get_verified_file_path(dataset_root)
        if os.path.exists(verified_file):
            try:
                with open(verified_file, 'r') as f:
                    self.verified_data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.verified_data = {}
        else:
            self.verified_data = {"verified_images": []}
    
    def save_verified_list(self, dataset_root):
        """Save verified images list to dataset."""
        metadata_dir = os.path.join(dataset_root, self.VERIFIED_METADATA_DIR)
        os.makedirs(metadata_dir, exist_ok=True)
        
        verified_file = self.get_verified_file_path(dataset_root)
        with open(verified_file, 'w') as f:
            json.dump(self.verified_data, f, indent=2)
    
    def mark_image_verified(self, dataset_root, image_path):
        """Mark an image as verified/cleaned.
        
        Args:
            dataset_root: Dataset root directory
            image_path: Path to the image file (can be absolute or relative)
        
        Returns:
            bool: True if marked, False if already marked
        """
        # Normalize path to be relative to dataset root
        if os.path.isabs(image_path):
            try:
                image_path = os.path.relpath(image_path, dataset_root)
            except ValueError:
                # Different drives on Windows
                pass
        
        # Normalize path separators for consistency
        image_path = str(image_path).replace("\\", "/")
        
        self.load_verified_list(dataset_root)
        
        if "verified_images" not in self.verified_data:
            self.verified_data["verified_images"] = []
        
        verified_list = self.verified_data["verified_images"]
        
        if image_path not in verified_list:
            verified_list.append(image_path)
            self.verified_data["last_verified"] = datetime.now().isoformat()
            self.save_verified_list(dataset_root)
            return True
        return False
    
    def is_verified(self, dataset_root, image_path):
        """Check if image has been verified.
        
        Args:
            dataset_root: Dataset root directory
            image_path: Path to the image file
        
        Returns:
            bool: True if verified, False otherwise
        """
        # Normalize path
        if os.path.isabs(image_path):
            try:
                image_path = os.path.relpath(image_path, dataset_root)
            except ValueError:
                pass
        
        image_path = str(image_path).replace("\\", "/")
        
        self.load_verified_list(dataset_root)
        verified_list = self.verified_data.get("verified_images", [])
        
        return image_path in verified_list
    
    def get_verified_count(self, dataset_root):
        """Get count of verified images."""
        self.load_verified_list(dataset_root)
        return len(self.verified_data.get("verified_images", []))
    
    def get_verified_images(self, dataset_root):
        """Get list of all verified images."""
        self.load_verified_list(dataset_root)
        return self.verified_data.get("verified_images", [])
    
    def clear_verified_list(self, dataset_root):
        """Clear all verified images (use with caution)."""
        self.verified_data = {"verified_images": []}
        self.save_verified_list(dataset_root)
    
    def remove_from_verified(self, dataset_root, image_path):
        """Remove an image from verified list."""
        # Normalize path
        if os.path.isabs(image_path):
            try:
                image_path = os.path.relpath(image_path, dataset_root)
            except ValueError:
                pass
        
        image_path = str(image_path).replace("\\", "/")
        
        self.load_verified_list(dataset_root)
        verified_list = self.verified_data.get("verified_images", [])
        
        if image_path in verified_list:
            verified_list.remove(image_path)
            self.save_verified_list(dataset_root)
            return True
        return False


def get_verified_images_for_scan(dataset_root):
    """Get set of verified image paths for filtering in Dataset Analysis.
    
    Args:
        dataset_root: Dataset root directory
    
    Returns:
        set: Set of normalized image paths that are verified
    """
    tracker = VerifiedImagesTracker()
    return set(tracker.get_verified_images(dataset_root))
