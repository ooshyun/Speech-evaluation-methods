#!/usr/bin/env python3
"""
Simple script to verify that the speech-evaluation-methods package is properly installed.
"""

def verify_installation():
    """Verify that the package can be imported and basic functions are available."""
    print("=" * 80)
    print("Verifying Speech Evaluation Methods Installation")
    print("=" * 80)
    
    try:
        import src
        print("✓ Package 'src' imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import 'src': {e}")
        return False
    
    # Check for main functions
    required_functions = [
        'estimate_stoi',
        'estimate_snr',
        'estimate_segsnr',
        'estimate_sdr',
        'estimate_si_sdr',
        'estimate_si_snr',
        'estimate_pesq',
        'estimate_composite',
        'estimate_mrsstftloss',
    ]
    
    missing_functions = []
    for func_name in required_functions:
        if hasattr(src, func_name):
            print(f"✓ Function '{func_name}' is available")
        else:
            print(f"✗ Function '{func_name}' is missing")
            missing_functions.append(func_name)
    
    if missing_functions:
        print(f"\n✗ Missing functions: {', '.join(missing_functions)}")
        return False
    
    # Check dependencies
    print("\n" + "=" * 80)
    print("Checking Dependencies")
    print("=" * 80)
    
    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'soundfile': 'soundfile',
        'librosa': 'librosa',
        'museval': 'museval',
        'pystoi': 'pystoi',
        'pesq': 'pesq',
        'torch': 'torch',
        'torchaudio': 'torchaudio',
        'torchmetrics': 'torchmetrics',
        'resampy': 'resampy',
        'auraloss': 'auraloss',
    }
    
    missing_deps = []
    for dep_name, module_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"✓ {dep_name} is installed")
        except ImportError:
            print(f"✗ {dep_name} is NOT installed")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"\n✗ Missing dependencies: {', '.join(missing_deps)}")
        print("\nTo install missing dependencies, run:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n" + "=" * 80)
    print("✓ All checks passed! The package is properly installed.")
    print("=" * 80)
    print("\nYou can now use the package by importing:")
    print("  from src import estimate_stoi, estimate_pesq, estimate_snr")
    print("\nOr run the test suite:")
    print("  python main.py")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    import sys
    success = verify_installation()
    sys.exit(0 if success else 1)

