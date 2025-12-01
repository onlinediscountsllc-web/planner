#!/usr/bin/env python3
"""
LIFE FRACTAL INTELLIGENCE - SIMPLE PYTHON DEPLOYER
Patches code and prepares for Render deployment
"""

import os
import sys
import shutil
from datetime import datetime

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def print_success(text):
    print(f"✓ {text}")

def print_error(text):
    print(f"✗ {text}")

def print_info(text):
    print(f"→ {text}")

def main():
    print_header("LIFE FRACTAL INTELLIGENCE - DEPLOYMENT")
    print("This will patch your code with all enhanced features")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Step 1: Check if we're in the right directory
    print_header("STEP 1: Checking Environment")
    
    if not (os.path.exists("life_planner_unified_master.py") or os.path.exists("life_fractal_render.py")):
        print_error("Main app file not found!")
        print_info("Please run this from your project directory")
        return
    
    main_file = "life_planner_unified_master.py" if os.path.exists("life_planner_unified_master.py") else "life_fractal_render.py"
    print_success(f"Found main file: {main_file}")
    
    # Step 2: Check for enhanced implementation
    if not os.path.exists("life_fractal_enhanced_implementation.py"):
        print_error("life_fractal_enhanced_implementation.py not found!")
        print_info("Please download it to this directory first")
        return
    
    print_success("Found enhanced implementation")
    
    # Step 3: Create backup
    print_header("STEP 2: Creating Backup")
    
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    files_to_backup = [main_file, "requirements.txt", "Procfile"]
    for file in files_to_backup:
        if os.path.exists(file):
            shutil.copy2(file, backup_dir)
            print_success(f"Backed up: {file}")
    
    print_info(f"Backup saved to: {backup_dir}/")
    
    # Step 4: Update requirements.txt
    print_header("STEP 3: Updating Requirements")
    
    requirements = """# Core Flask
flask==3.0.0
flask-cors==4.0.0
werkzeug==3.0.0

# Data Processing
numpy==1.24.0
pillow==10.0.0

# Security
pyjwt==2.8.0
bcrypt==4.0.1

# Payments
stripe==5.5.0

# Server
gunicorn==21.2.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print_success("Updated requirements.txt")
    
    # Step 5: Update/create Procfile
    print_header("STEP 4: Updating Procfile")
    
    if "life_fractal_render.py" in main_file:
        procfile = "web: gunicorn life_fractal_render:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120"
    else:
        procfile = "web: gunicorn life_planner_unified_master:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120"
    
    with open("Procfile", "w") as f:
        f.write(procfile)
    
    print_success("Updated Procfile")
    
    # Step 6: Update/create runtime.txt
    print_header("STEP 5: Setting Python Version")
    
    with open("runtime.txt", "w") as f:
        f.write("python-3.11.6")
    
    print_success("Set Python version to 3.11.6")
    
    # Step 7: Patch main file
    print_header("STEP 6: Patching Main Application")
    
    with open(main_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if already patched
    if "from life_fractal_enhanced_implementation import" in content:
        print_info("Already patched - skipping")
    else:
        # Find where to insert imports
        import_patch = """
# Enhanced features import
try:
    from life_fractal_enhanced_implementation import (
        EmotionalPetAI,
        FractalTimeCalendar,
        FibonacciTaskScheduler,
        ExecutiveFunctionSupport,
        AutismSafeColors,
        AphantasiaSupport,
        PrivacyPreservingML
    )
    ENHANCED_FEATURES_AVAILABLE = True
    print("Enhanced features loaded successfully")
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    print(f"Enhanced features not available: {e}")
"""
        
        # Insert after Flask imports
        if "from flask import" in content:
            insert_point = content.find("from flask import")
            # Find next blank line after imports
            next_newline = content.find("\n\n", insert_point)
            if next_newline > 0:
                content = content[:next_newline] + "\n" + import_patch + content[next_newline:]
                print_success("Added enhanced features import")
            else:
                print_error("Could not find insertion point")
                print_info("You may need to manually add the import")
        else:
            print_error("Could not find Flask imports")
            print_info("You may need to manually add the import")
        
        # Save patched file
        with open(main_file, "w", encoding="utf-8") as f:
            f.write(content)
    
    # Step 8: Final summary
    print_header("DEPLOYMENT READY!")
    
    print("Changes made:")
    print_success("Enhanced implementation file ready")
    print_success("Requirements updated")
    print_success("Procfile configured")
    print_success("Runtime set to Python 3.11.6")
    print_success(f"Main file patched: {main_file}")
    print()
    
    print("Next steps:")
    print_info("1. Commit changes: git add . && git commit -m 'feat: Add enhanced features'")
    print_info("2. Push to Render: git push render main")
    print_info("3. Monitor deployment at: https://dashboard.render.com/")
    print()
    
    print_header("SUCCESS!")
    print("Your Life Fractal Intelligence is ready for deployment!")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDeployment cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        sys.exit(1)
