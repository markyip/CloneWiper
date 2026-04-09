
import os
import sys
import time
import shutil

# Ensure we can import core modules
sys.path.append(os.getcwd())

from core.thumbnail_cache import PersistentThumbnailCache

def verify_cache():
    print("Testing PersistentThumbnailCache...")
    
    # Use a temporary DB file
    db_path = "test_thumbnails.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    cache = PersistentThumbnailCache(db_path=db_path)
    
    # Test Data
    test_path = os.path.abspath("test_image.jpg")
    test_mtime = 123456789
    test_size = 1024
    test_width = 100
    test_height = 100
    test_data = b"fake_jpeg_data"
    
    # 1. Put
    print("1. Testing PUT...")
    cache.put(test_path, test_mtime, test_size, test_width, test_height, test_data)
    
    # 2. Get (Hit)
    print("2. Testing GET (Hit)...")
    retrieved = cache.get(test_path, test_mtime, test_size)
    if retrieved == test_data:
        print("   PASS: Data retrieved correctly.")
    else:
        print(f"   FAIL: Retrieved data mismatch. Got {retrieved}")
        return False
        
    # 3. Get (Miss - wrong mtime)
    print("3. Testing GET (Miss - Modified file)...")
    miss = cache.get(test_path, test_mtime + 1, test_size)
    if miss is None:
        print("   PASS: Correctly returned None for modified mtime.")
    else:
        print("   FAIL: Should have missed.")
        return False

    # 4. Get (Miss - wrong size)
    print("4. Testing GET (Miss - Resized file)...")
    miss = cache.get(test_path, test_mtime, test_size + 1)
    if miss is None:
        print("   PASS: Correctly returned None for modified size.")
    else:
        print("   FAIL: Should have missed.")
        return False

    # 5. Persistence (Close and Reopen)
    print("5. Testing Persistence...")
    del cache
    cache2 = PersistentThumbnailCache(db_path=db_path)
    retrieved2 = cache2.get(test_path, test_mtime, test_size)
    if retrieved2 == test_data:
        print("   PASS: Data persisted after reload.")
    else:
        print("   FAIL: Persistence failed.")
        return False
        
    # Cleanup
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except:
            pass
            
    print("\nSUCCESS: PersistentThumbnailCache verified!")
    return True

if __name__ == "__main__":
    success = verify_cache()
    sys.exit(0 if success else 1)
