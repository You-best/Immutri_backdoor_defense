"""
Quick verification script to check IMMUTRI code structure and imports.
This does NOT run the full pipeline, just verifies that all components are accessible.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def verify_imports():
    """Verify all critical imports work correctly."""
    print("=" * 80)
    print("IMMUTRI Code Structure Verification")
    print("=" * 80)
    
    errors = []
    
    # Test core module imports
    print("\n1. Testing core module imports...")
    try:
        from core.feature_analyzer import (
            analyze_and_partition_dataset,
            detect_first_minimum_loss,
            infer_target_label,
            directional_feature_matching,
            get_losses_and_data,
            extract_features,
            calculate_class_prototypes
        )
        print("   ✓ feature_analyzer.py - All functions imported successfully")
    except Exception as e:
        errors.append(f"feature_analyzer.py import error: {str(e)}")
        print(f"   ✗ feature_analyzer.py - ERROR: {e}")
    
    try:
        from core.data_purification import (
            train_auxiliary_detector_with_uniform_noise,
            purify_dataset_with_detector,
            train_reference_model,
            tri_model_firewall_inference,
            get_filtered_data
        )
        print("   ✓ data_purification.py - All functions imported successfully")
    except Exception as e:
        errors.append(f"data_purification.py import error: {str(e)}")
        print(f"   ✗ data_purification.py - ERROR: {e}")
    
    # Test utility imports
    print("\n2. Testing utility module imports...")
    try:
        from utils.initial import get_task, applied_attack, get_dataloader
        from utils.utils import evaluate, evaluate_model, CustomDataset
        from utils.parameters import My_Params, set_param
        print("   ✓ utils modules - All imports successful")
    except Exception as e:
        errors.append(f"utils import error: {str(e)}")
        print(f"   ✗ utils modules - ERROR: {e}")
    
    # Test task imports
    print("\n3. Testing task module imports...")
    try:
        from tasks.cifar10_task import Cifar10Task
        from tasks.gtsrb_task import GtsrbTask
        from tasks.imagenet10_task import Imagenet10Task
        print("   ✓ task modules - All imports successful")
    except Exception as e:
        errors.append(f"task import error: {str(e)}")
        print(f"   ✗ task modules - ERROR: {e}")
    
    # Test attack imports
    print("\n4. Testing attack module imports...")
    try:
        from attacks.badnets_synthesizer import BadnetsSynthesizer
        from attacks.blend_synthesizer import BlendSynthesizer
        from attacks.CL_synthesizer import CLSynthesizer
        print("   ✓ attack modules - All imports successful")
    except Exception as e:
        errors.append(f"attack import error: {str(e)}")
        print(f"   ✗ attack modules - ERROR: {e}")
    
    # Verify function signatures
    print("\n5. Verifying function signatures...")
    import inspect
    
    # Check LGFD functions
    sig = inspect.signature(analyze_and_partition_dataset)
    params = list(sig.parameters.keys())
    if params == ['model', 'dataloader', 'param']:
        print("   ✓ analyze_and_partition_dataset signature correct")
    else:
        errors.append("analyze_and_partition_dataset has wrong signature")
        print(f"   ✗ analyze_and_partition_dataset signature incorrect: {params}")
    
    sig = inspect.signature(infer_target_label)
    params = list(sig.parameters.keys())
    expected = ['model', 'dataloader', 'suspicious_indices', 'device', 'num_classes']
    if params == expected:
        print("   ✓ infer_target_label signature correct")
    else:
        errors.append("infer_target_label has wrong signature")
        print(f"   ✗ infer_target_label signature incorrect: {params}")
    
    # Check TMFS functions
    sig = inspect.signature(train_auxiliary_detector_with_uniform_noise)
    params = list(sig.parameters.keys())
    expected = ['model', 'dataloader', 'param', 'num_epochs', 'lr', 'noise_rate']
    if params == expected:
        print("   ✓ train_auxiliary_detector_with_uniform_noise signature correct")
    else:
        errors.append("train_auxiliary_detector_with_uniform_noise has wrong signature")
        print(f"   ✗ train_auxiliary_detector_with_uniform_noise signature incorrect: {params}")
    
    sig = inspect.signature(tri_model_firewall_inference)
    params = list(sig.parameters.keys())
    expected = ['fbd', 'fenh', 'fref', 'input_tensor', 'param', 'threshold']
    if params == expected:
        print("   ✓ tri_model_firewall_inference signature correct")
    else:
        errors.append("tri_model_firewall_inference has wrong signature")
        print(f"   ✗ tri_model_firewall_inference signature incorrect: {params}")
    
    # Summary
    print("\n" + "=" * 80)
    if errors:
        print(f"VERIFICATION FAILED - {len(errors)} error(s) found:")
        for error in errors:
            print(f"  • {error}")
        return False
    else:
        print("✅ VERIFICATION SUCCESSFUL - All components are properly structured!")
        print("\nThe IMMUTRI implementation is ready for use.")
        print("\nNext steps:")
        print("  1. Prepare your dataset (CIFAR10/GTSRB/ImageNet10)")
        print("  2. Train a backdoored model: python attack.py --dataset cifar10 --attack badnets")
        print("  3. Run IMMUTRI defense: python ImmuTri.py --dataset cifar10 --attack badnets")
        return True

if __name__ == '__main__':
    success = verify_imports()
    sys.exit(0 if success else 1)
