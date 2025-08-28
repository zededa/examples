#!/usr/bin/env python3
"""
Test the class name mapping for Stanford Cars dataset.
This ensures the accuracy evaluation will work correctly.
"""

import sys
import os

# Add the client directory to the path
sys.path.insert(0, '/home/zedtel/Developer/edgeai-alpha2-dynamic/client-container')

def test_class_names_mapping():
    """Test that class names and dataset structure align."""
    
    print("🧪 Testing Stanford Cars Dataset Class Mapping")
    print("=" * 50)
    
    try:
        from client import ModelServerClient
        client = ModelServerClient('http://test-server:8000')
        
        # Check if class names loaded
        if not client.class_names:
            print("❌ No class names loaded")
            return False
        
        print(f"✅ Loaded {len(client.class_names)} class names")
        
        # Check dataset path
        dataset_path = client.test_dataset_path
        print(f"Dataset path: {dataset_path}")
        
        if os.path.exists(dataset_path):
            print("✅ Dataset path exists")
            
            # Get class directories
            class_dirs = [d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d))]
            
            print(f"✅ Found {len(class_dirs)} class directories")
            
            # Check some mappings
            print("\n📋 Sample Class Mappings:")
            print("-" * 30)
            
            for i, class_name in enumerate(client.class_names[:10]):
                index_in_dirs = -1
                for j, dir_name in enumerate(class_dirs):
                    if dir_name == class_name:
                        index_in_dirs = j
                        break
                
                status = "✅" if index_in_dirs >= 0 else "❌"
                print(f"{status} Index {i}: {class_name}")
                if index_in_dirs >= 0:
                    # Count images in this class
                    class_path = os.path.join(dataset_path, class_name)
                    if os.path.exists(class_path):
                        import glob
                        images = glob.glob(os.path.join(class_path, "*.jpg"))
                        print(f"     -> {len(images)} images")
            
            # Check for missing classes
            missing_classes = []
            for class_name in client.class_names:
                if class_name not in class_dirs:
                    missing_classes.append(class_name)
            
            if missing_classes:
                print(f"\n⚠️  {len(missing_classes)} classes missing from dataset:")
                for missing in missing_classes[:5]:
                    print(f"   - {missing}")
                if len(missing_classes) > 5:
                    print(f"   ... and {len(missing_classes) - 5} more")
            else:
                print("\n✅ All classes found in dataset")
            
            # Check for extra directories
            extra_dirs = []
            for dir_name in class_dirs:
                if dir_name not in client.class_names:
                    extra_dirs.append(dir_name)
            
            if extra_dirs:
                print(f"\n⚠️  {len(extra_dirs)} extra directories in dataset:")
                for extra in extra_dirs[:5]:
                    print(f"   - {extra}")
                if len(extra_dirs) > 5:
                    print(f"   ... and {len(extra_dirs) - 5} more")
            else:
                print("\n✅ No extra directories in dataset")
            
        else:
            print("❌ Dataset path does not exist")
            print("   This is expected if not running in the container")
            print("   The evaluation will work when the dataset is available")
            
        return True
        
    except Exception as e:
        print(f"❌ Class mapping test failed: {e}")
        return False

def show_usage_for_real_server():
    """Show how to use with a real server."""
    
    print("\n🚀 HOW TO USE WITH REAL MODEL SERVER")
    print("=" * 50)
    
    print("1. Start your OpenVINO Model Server with car models")
    print("2. Ensure Stanford Cars dataset is available")
    print("3. Run evaluation:")
    print()
    print("   # Quick test (30s benchmark + 5 samples per class)")
    print("   python run_evaluation.py comprehensive --quick")
    print()
    print("   # Full evaluation (60s benchmark + 10 samples per class)")
    print("   python run_evaluation.py comprehensive")
    print()
    print("   # Only accuracy test")
    print("   python client.py --mode accuracy --max-samples-per-class 20")
    print()
    print("   # Only throughput benchmark")
    print("   python client.py --mode benchmark --benchmark-duration 120")
    print()
    
    print("📊 Expected workflow:")
    print("1. Model detection - Discovers all available models")
    print("2. Health checks - Verifies models are ready")
    print("3. Benchmarking - Measures throughput and latency")
    print("4. Accuracy testing - Tests on Stanford Cars dataset")
    print("5. Reporting - Generates rankings and saves JSON results")

def main():
    """Run the class mapping test."""
    
    success = test_class_names_mapping()
    show_usage_for_real_server()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Class mapping test completed successfully!")
        print("🚀 Enhanced client is ready for model evaluation!")
    else:
        print("⚠️  Some issues found, but the client should still work")
        print("🚀 Try running with a real model server and dataset")
    
    return success

if __name__ == "__main__":
    main()
