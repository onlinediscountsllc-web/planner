"""
Life Planner - Database Initialization (Self-Healing)
Automatically handles errors and creates database
"""

import os
import sys
from pathlib import Path

# Ensure we're in the right directory
os.chdir(Path(__file__).parent)

print("=" * 70)
print("LIFE PLANNER - DATABASE SETUP")
print("=" * 70)
print()

# ============================================================================
# Load environment
# ============================================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ Loaded .env file")
except:
    print("⚠ python-dotenv not available, using defaults")

print()

# ============================================================================
# Import Flask app with error handling
# ============================================================================

try:
    # Try refactored app first
    if os.path.exists('app_refactored.py'):
        print("Loading refactored application...")
        from app_refactored import app, db, User
    else:
        print("Loading standard application...")
        from app import app, db
        from models.database import User
    
    print("✓ Application loaded")
except Exception as e:
    print(f"✗ Error loading application: {e}")
    print()
    print("SOLUTION:")
    print("1. Make sure Flask is installed: pip install flask flask-sqlalchemy")
    print("2. Run setup_and_fix.py first")
    print("3. Check that app.py exists")
    sys.exit(1)

print()

# ============================================================================
# Initialize Database
# ============================================================================

def init_database():
    """Create all database tables"""
    try:
        with app.app_context():
            print("Creating database tables...")
            db.create_all()
            print("✓ Database tables created")
            print()
            
            # Create admin user
            admin_email = os.getenv('ADMIN_EMAIL', 'onlinediscountsllc@gmail.com')
            admin_password = os.getenv('ADMIN_PASSWORD', 'admin8587037321')
            
            admin = User.query.filter_by(email=admin_email).first()
            
            if not admin:
                print("Creating admin user...")
                
                try:
                    from werkzeug.security import generate_password_hash
                    
                    admin = User()
                    admin.email = admin_email
                    admin.password_hash = generate_password_hash(admin_password)
                    
                    # Set admin attributes if they exist
                    if hasattr(admin, 'is_admin'):
                        admin.is_admin = True
                    if hasattr(admin, 'subscription_status'):
                        admin.subscription_status = 'active'
                    if hasattr(admin, 'email_verified'):
                        admin.email_verified = True
                    
                    db.session.add(admin)
                    db.session.commit()
                    
                    print(f"✓ Admin user created: {admin_email}")
                    print(f"  Password: {admin_password}")
                    print()
                    print("  ⚠ CHANGE THIS PASSWORD AFTER FIRST LOGIN!")
                    
                except Exception as e:
                    print(f"⚠ Could not create admin user: {e}")
                    print("  You can create users through registration")
            else:
                print(f"✓ Admin user already exists: {admin_email}")
            
            print()
            
            # Show statistics
            try:
                total_users = User.query.count()
                print(f"Total users in database: {total_users}")
            except:
                print("Database initialized (user count unavailable)")
            
            print()
            print("=" * 70)
            print("DATABASE INITIALIZATION COMPLETE!")
            print("=" * 70)
            print()
            print("NEXT STEPS:")
            print("1. Edit .env file with your Stripe and Gmail credentials")
            print("2. Run: python app.py  (or app_refactored.py)")
            print("3. Open: http://localhost:5000")
            print("4. Login with admin credentials above")
            print()
            print("=" * 70)
            
            return True
            
    except Exception as e:
        print(f"✗ Database initialization failed: {e}")
        print()
        print("TROUBLESHOOTING:")
        print("- Check that you have write permissions in this directory")
        print("- Make sure no other app is using the database file")
        print("- Try deleting life_planner.db and running again")
        return False

def show_stats():
    """Show database statistics"""
    try:
        with app.app_context():
            print("=" * 70)
            print("DATABASE STATISTICS")
            print("=" * 70)
            print()
            
            total_users = User.query.count()
            print(f"Total Users: {total_users}")
            
            try:
                from models.database import Pet, UserActivity
                total_pets = Pet.query.count()
                total_activities = UserActivity.query.count()
                print(f"Total Pets: {total_pets}")
                print(f"Total Activities: {total_activities}")
            except:
                print("(Extended stats not available)")
            
            print()
            
            # Recent users
            recent = User.query.order_by(User.id.desc()).limit(5).all()
            if recent:
                print("Recent Users:")
                for user in recent:
                    print(f"  - {user.email}")
            
            print()
            print("=" * 70)
            
    except Exception as e:
        print(f"Could not retrieve statistics: {e}")

def reset_database():
    """Reset database (DANGEROUS!)"""
    print()
    print("⚠" * 35)
    print("WARNING: THIS WILL DELETE ALL DATA!")
    print("⚠" * 35)
    print()
    confirm = input("Type 'DELETE ALL DATA' to confirm: ")
    
    if confirm == 'DELETE ALL DATA':
        try:
            with app.app_context():
                print("Dropping all tables...")
                db.drop_all()
                print("✓ All tables dropped")
                print()
                print("Recreating database...")
                init_database()
        except Exception as e:
            print(f"Reset failed: {e}")
    else:
        print("Reset cancelled")

# ============================================================================
# Interactive Menu
# ============================================================================

def main_menu():
    """Interactive database management"""
    while True:
        print()
        print("=" * 70)
        print("DATABASE MANAGEMENT MENU")
        print("=" * 70)
        print()
        print("1. Initialize Database (first time setup)")
        print("2. Show Statistics")
        print("3. Reset Database (⚠ DANGER - deletes everything)")
        print("4. Exit")
        print()
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            init_database()
        elif choice == '2':
            show_stats()
        elif choice == '3':
            reset_database()
        elif choice == '4':
            print()
            print("Goodbye!")
            break
        else:
            print("Invalid choice")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == '__main__':
    # Check if running with argument
    if len(sys.argv) > 1:
        if sys.argv[1] == '--auto' or sys.argv[1] == '-a':
            # Auto mode: just initialize and exit
            init_database()
        elif sys.argv[1] == '--stats' or sys.argv[1] == '-s':
            show_stats()
        else:
            print("Usage: python init_db.py [--auto|--stats]")
    else:
        # Interactive mode
        try:
            main_menu()
        except KeyboardInterrupt:
            print()
            print("Interrupted by user")
        except Exception as e:
            print(f"Error: {e}")
