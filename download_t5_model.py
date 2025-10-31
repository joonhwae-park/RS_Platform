#!/usr/bin/env python3
"""
Download t5-small model files for Docker image.
Run this script before building the Docker image.
"""
import os
import shutil
from transformers import T5Tokenizer, T5ForConditionalGeneration

def download_t5_small():
    print("=" * 60)
    print("Downloading t5-small model files...")
    print("=" * 60)

    # Create temporary cache directory
    cache_dir = "./huggingface_cache"
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Download model and tokenizer
        print("\n1. Downloading model weights...")
        model = T5ForConditionalGeneration.from_pretrained(
            "google-t5/t5-small",
            cache_dir=cache_dir
        )
        print("   ✓ Model weights downloaded")

        print("\n2. Downloading tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(
            "google-t5/t5-small",
            cache_dir=cache_dir
        )
        print("   ✓ Tokenizer downloaded")

        # Find the snapshot directory
        models_dir = os.path.join(cache_dir, "models--google-t5--t5-small")
        snapshots_dir = os.path.join(models_dir, "snapshots")

        if not os.path.exists(snapshots_dir):
            raise Exception(f"Snapshots directory not found: {snapshots_dir}")

        # Get the first (and usually only) snapshot hash directory
        snapshot_dirs = [d for d in os.listdir(snapshots_dir)
                        if os.path.isdir(os.path.join(snapshots_dir, d))]

        if not snapshot_dirs:
            raise Exception("No snapshot directories found")

        source_dir = os.path.join(snapshots_dir, snapshot_dirs[0])

        # Create target directory
        target_dir = "./models/t5-small"
        os.makedirs(target_dir, exist_ok=True)

        # Copy files
        print(f"\n3. Copying files from cache to {target_dir}...")
        for item in os.listdir(source_dir):
            source = os.path.join(source_dir, item)
            target = os.path.join(target_dir, item)

            if os.path.isfile(source):
                shutil.copy2(source, target)
                size_mb = os.path.getsize(source) / (1024 * 1024)
                print(f"   ✓ {item} ({size_mb:.2f} MB)")
            elif os.path.isdir(source):
                shutil.copytree(source, target, dirs_exist_ok=True)
                print(f"   ✓ {item}/ (directory)")

        # Clean up cache
        print("\n4. Cleaning up temporary cache...")
        shutil.rmtree(cache_dir)
        print("   ✓ Cache removed")

        print("\n" + "=" * 60)
        print("SUCCESS! Model files ready at ./models/t5-small/")
        print("You can now build the Docker image.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have transformers and torch installed:")
        print("  pip install transformers torch")
        return False

    return True

if __name__ == "__main__":
    success = download_t5_small()
    exit(0 if success else 1)
