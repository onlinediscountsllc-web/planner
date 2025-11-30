"""
Database Initialization Script
Sets up database schema, creates admin user, and initializes system settings
"""

import os
import sys
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db
from models.database import User, Pet, SystemSettings, AuditLog

def init_database():
    """Initialize database with tables and default data"""
    
    print("🔧 Initializing Life Planner Database...")
    print("=" * 60)
    
    with app.app_context():
        # Create all tables
        print("\n📊 Creating database tables...")
        db.create_all()
        print("✅ Tables created successfully")
        
        # Create admin user
        print("\n👤 Setting up admin user...")
        admin_email = os.getenv('ADMIN_EMAIL', 'onlinediscountsllc@gmail.com')
        admin = User.query.filter_by(email=admin_email).first()
        
        if not admin:
            admin = User(
                email=admin_email,
                first_name='Luke',
                last_name='Smith',
                is_admin=True,
                is_active=True,
                email_verified=True,
                subscription_status='active',
                subscription_start_date=datetime.utcnow(),
                subscription_end_date=datetime.utcnow() + timedelta(days=365)
            )
            admin.set_password(os.getenv('ADMIN_PASSWORD', 'admin8587037321'))
            db.session.add(admin)
            
            # Create admin's pet
            admin_pet = Pet(
                user_id=admin.id,
                name='Dragon Master',
                species='dragon',
                level=50,
                experience=10000,
                mood=100,
                energy=100,
                hunger=0,
                stress=0,
                growth=100,
                bond=100,
                behavior='happy'
            )
            db.session.add(admin_pet)
            
            print(f"✅ Admin user created: {admin_email}")
            print(f"   Password: admin8587037321 (CHANGE THIS IMMEDIATELY!)")
        else:
            print(f"ℹ️  Admin user already exists: {admin_email}")
        
        # Initialize system settings
        print("\n⚙️  Setting up system configuration...")
        settings = [
            {
                'key': 'subscription_price',
                'value': '20.00',
                'description': 'Monthly subscription price in USD'
            },
            {
                'key': 'trial_days',
                'value': '7',
                'description': 'Number of days for free trial'
            },
            {
                'key': 'gofundme_url',
                'value': os.getenv('GOFUNDME_URL', 'https://gofund.me/8d9303d27'),
                'description': 'GoFundMe campaign URL'
            },
            {
                'key': 'max_free_users',
                'value': '1000',
                'description': 'Maximum number of concurrent trial users'
            },
            {
                'key': 'ml_model_version',
                'value': '1.0.0',
                'description': 'Current ML model version'
            },
            {
                'key': 'feature_flags',
                'value': 'gpu_acceleration:true,federated_learning:true,ancient_math:true',
                'description': 'Enabled feature flags'
            }
        ]
        
        for setting in settings:
            existing = SystemSettings.query.filter_by(key=setting['key']).first()
            if not existing:
                new_setting = SystemSettings(**setting)
                db.session.add(new_setting)
                print(f"   ✓ {setting['key']}: {setting['value']}")
            else:
                print(f"   ℹ️  {setting['key']}: Already exists")
        
        # Commit all changes
        print("\n💾 Saving changes to database...")
        db.session.commit()
        print("✅ Database initialization complete!")
        
        # Display summary
        print("\n" + "=" * 60)
        print("📊 DATABASE SUMMARY")
        print("=" * 60)
        
        total_users = User.query.count()
        active_subs = User.query.filter_by(subscription_status='active').count()
        trial_users = User.query.filter_by(subscription_status='trial').count()
        total_pets = Pet.query.count()
        
        print(f"Total Users:         {total_users}")
        print(f"Active Subscriptions: {active_subs}")
        print(f"Trial Users:         {trial_users}")
        print(f"Virtual Pets:        {total_pets}")
        print(f"Admin Email:         {admin_email}")
        print("\n✨ Database is ready for production!")
        print("=" * 60)


def create_test_users():
    """Create test users for development/testing"""
    print("\n🧪 Creating test users...")
    
    with app.app_context():
        test_users = [
            {
                'email': 'test1@example.com',
                'password': 'testpass123',
                'first_name': 'Test',
                'last_name': 'User1',
                'species': 'cat'
            },
            {
                'email': 'test2@example.com',
                'password': 'testpass123',
                'first_name': 'Test',
                'last_name': 'User2',
                'species': 'dragon'
            },
            {
                'email': 'test3@example.com',
                'password': 'testpass123',
                'first_name': 'Test',
                'last_name': 'User3',
                'species': 'phoenix'
            }
        ]
        
        for user_data in test_users:
            existing = User.query.filter_by(email=user_data['email']).first()
            if not existing:
                user = User(
                    email=user_data['email'],
                    first_name=user_data['first_name'],
                    last_name=user_data['last_name'],
                    email_verified=True,
                    subscription_status='trial',
                    trial_start_date=datetime.utcnow(),
                    trial_end_date=datetime.utcnow() + timedelta(days=7)
                )
                user.set_password(user_data['password'])
                db.session.add(user)
                db.session.flush()
                
                # Create pet
                pet = Pet(
                    user_id=user.id,
                    name=f"{user_data['first_name']}'s Pet",
                    species=user_data['species']
                )
                db.session.add(pet)
                
                print(f"✅ Created: {user_data['email']} | Password: testpass123")
            else:
                print(f"ℹ️  Already exists: {user_data['email']}")
        
        db.session.commit()
        print("✅ Test users created")


def reset_database():
    """⚠️ WARNING: Drops all tables and recreates them"""
    print("\n⚠️  WARNING: This will DELETE ALL DATA!")
    confirm = input("Type 'DELETE ALL DATA' to confirm: ")
    
    if confirm == 'DELETE ALL DATA':
        print("\n🗑️  Dropping all tables...")
        with app.app_context():
            db.drop_all()
            print("✅ All tables dropped")
        
        print("\n🔧 Recreating database...")
        init_database()
    else:
        print("❌ Reset cancelled")


def show_stats():
    """Display database statistics"""
    with app.app_context():
        print("\n" + "=" * 60)
        print("📊 CURRENT DATABASE STATISTICS")
        print("=" * 60)
        
        # User stats
        total_users = User.query.count()
        active_subs = User.query.filter_by(subscription_status='active').count()
        trial_users = User.query.filter_by(subscription_status='trial').count()
        cancelled = User.query.filter_by(subscription_status='cancelled').count()
        
        print(f"\n👥 Users:")
        print(f"   Total:              {total_users}")
        print(f"   Active Paid:        {active_subs}")
        print(f"   Trial:              {trial_users}")
        print(f"   Cancelled:          {cancelled}")
        
        # Pet stats
        from sqlalchemy import func
        species_count = db.session.query(
            Pet.species, 
            func.count(Pet.species)
        ).group_by(Pet.species).all()
        
        print(f"\n🐾 Pets:")
        print(f"   Total:              {Pet.query.count()}")
        for species, count in species_count:
            print(f"   {species.capitalize():12}    {count}")
        
        # Activity stats
        from models.database import UserActivity
        total_activities = UserActivity.query.count()
        today = datetime.utcnow().date()
        today_activities = UserActivity.query.filter(
            UserActivity.timestamp >= today
        ).count()
        
        print(f"\n📈 Activity:")
        print(f"   Total Entries:      {total_activities}")
        print(f"   Today:              {today_activities}")
        
        # Revenue estimation
        monthly_revenue = active_subs * 20.00
        print(f"\n💰 Revenue (Estimated):")
        print(f"   Monthly:            ${monthly_revenue:.2f}")
        print(f"   Annual:             ${monthly_revenue * 12:.2f}")
        
        # Recent users
        recent = User.query.order_by(User.created_at.desc()).limit(5).all()
        print(f"\n🆕 Recent Users:")
        for user in recent:
            print(f"   {user.email:30} {user.created_at.strftime('%Y-%m-%d %H:%M')}")
        
        print("\n" + "=" * 60)


def main():
    """Main menu for database management"""
    while True:
        print("\n" + "=" * 60)
        print("LIFE PLANNER - DATABASE MANAGEMENT")
        print("=" * 60)
        print("\n1. Initialize Database (first time setup)")
        print("2. Create Test Users")
        print("3. Show Statistics")
        print("4. Reset Database (⚠️  DANGER)")
        print("5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            init_database()
        elif choice == '2':
            create_test_users()
        elif choice == '3':
            show_stats()
        elif choice == '4':
            reset_database()
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")


if __name__ == '__main__':
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  WARNING: .env file not found!")
        print("Please copy .env.template to .env and configure it.")
        sys.exit(1)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run main menu
    main()
