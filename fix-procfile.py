#!/usr/bin/env python3
"""
ULTRA-SIMPLE FIX FOR APP.PY IMPORT ERROR
Just update the Procfile to use the correct main file
"""

import os

def print_header(text):
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def print_success(text):
    print(f"[OK] {text}")

def print_info(text):
    print(f">>> {text}")

def main():
    print_header("FIX app.py IMPORT ERROR")
    
    # Check which main files exist
    main_files = []
    for f in ['app.py', 'life_planner_unified_master.py', 'life_fractal_render.py']:
        if os.path.exists(f):
            main_files.append(f)
            print_success(f"Found: {f}")
    
    if not main_files:
        print("[ERROR] No main Python file found!")
        return
    
    print()
    
    # Determine which file to use
    if 'life_planner_unified_master.py' in main_files:
        main_file = 'life_planner_unified_master'
        print_info(f"Will use: {main_file}.py")
    elif 'life_fractal_render.py' in main_files:
        main_file = 'life_fractal_render'
        print_info(f"Will use: {main_file}.py")
    else:
        # Check if app.py has numpy
        if os.path.exists('app.py'):
            with open('app.py', 'r') as f:
                content = f.read()
            
            if 'import numpy' in content:
                print("[ERROR] app.py imports numpy and no other main file found!")
                print()
                print("Solutions:")
                print("  1. Delete app.py if it's old/unused")
                print("  2. Remove 'import numpy' from app.py manually")
                print("  3. Use life_planner_unified_master.py or life_fractal_render.py")
                return
            else:
                main_file = 'app'
                print_info("Will use: app.py")
    
    # Update Procfile
    print()
    print_header("UPDATING PROCFILE")
    
    procfile_content = f"web: gunicorn {main_file}:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120"
    
    with open('Procfile', 'w') as f:
        f.write(procfile_content)
    
    print_success("Updated Procfile")
    print_info(f"Command: gunicorn {main_file}:app")
    
    # Also check if app.py should be deleted
    print()
    if os.path.exists('app.py') and main_file != 'app':
        print_header("OPTIONAL: CLEAN UP")
        print_info("You have an app.py file but using a different main file")
        print_info("You can delete app.py if it's old/unused:")
        print()
        print("  rm app.py")
        print("  # or")
        print("  del app.py")
        print()
    
    print_header("DEPLOY NOW!")
    print("Run these commands:")
    print()
    print("  git add Procfile")
    print("  git commit -m 'fix: Update Procfile to use correct main file'")
    print("  git push origin main")
    print()
    print("Your app will now start successfully!")
    print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\n[ERROR]: {e}")
