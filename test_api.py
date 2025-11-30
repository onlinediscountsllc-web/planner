"""
🧪 Life Fractal Intelligence - API Test Suite
═══════════════════════════════════════════════════════════════════════════════════════
Tests all API endpoints to ensure production readiness
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:5000"
TEST_EMAIL = f"test_{datetime.now().timestamp()}@example.com"
TEST_PASSWORD = "TestPassword123!"
TOKEN = None


def log_test(name: str, passed: bool, details: str = ""):
    """Log test result"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {name}")
    if details:
        print(f"    {details}")


def test_health_check():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            details = f"Users: {data['users']}, GPU: {data['gpu']}"
        else:
            details = f"Status code: {response.status_code}"
        
        log_test("Health Check", passed, details)
        return passed
    except Exception as e:
        log_test("Health Check", False, str(e))
        return False


def test_register():
    """Test user registration"""
    global TOKEN
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/register",
            json={
                "email": TEST_EMAIL,
                "password": TEST_PASSWORD,
                "pet_species": "dragon",
                "pet_name": "TestDragon"
            }
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            TOKEN = data['token']
            details = f"User: {data['user']['email']}, Pet: {data['user']['pet']['name']}"
        else:
            details = f"Status: {response.status_code}, Error: {response.text}"
        
        log_test("User Registration", passed, details)
        return passed
    except Exception as e:
        log_test("User Registration", False, str(e))
        return False


def test_login():
    """Test user login"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/login",
            json={
                "email": TEST_EMAIL,
                "password": TEST_PASSWORD
            }
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            details = f"Token received, User: {data['user']['email']}"
        else:
            details = f"Status: {response.status_code}"
        
        log_test("User Login", passed, details)
        return passed
    except Exception as e:
        log_test("User Login", False, str(e))
        return False


def test_create_goal():
    """Test goal creation"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/goals",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "title": "Test Goal",
                "description": "This is a test goal",
                "category": "testing",
                "priority": "high",
                "target_date": "2025-12-31"
            }
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            details = f"Goal created: {data['goal']['title']}"
            return passed, data['goal']['id']
        else:
            details = f"Status: {response.status_code}"
            log_test("Create Goal", passed, details)
            return passed, None
        
        log_test("Create Goal", passed, details)
        return passed, None
    except Exception as e:
        log_test("Create Goal", False, str(e))
        return False, None


def test_add_task(goal_id: str):
    """Test task creation"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/goals/{goal_id}/tasks",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "title": "Test Task",
                "priority": "medium",
                "estimated_hours": 2.0
            }
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            details = f"Task created: {data['task']['title']}"
            return passed, data['task']['id']
        else:
            details = f"Status: {response.status_code}"
            log_test("Add Task", passed, details)
            return passed, None
        
    except Exception as e:
        log_test("Add Task", False, str(e))
        return False, None


def test_complete_task(task_id: str):
    """Test task completion"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/tasks/{task_id}/complete",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            details = f"XP gained: {data['xp_gained']}, Pet level: {data['pet']['level']}"
        else:
            details = f"Status: {response.status_code}"
        
        log_test("Complete Task", passed, details)
        return passed
    except Exception as e:
        log_test("Complete Task", False, str(e))
        return False


def test_create_habit():
    """Test habit creation"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/habits",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "title": "Daily Exercise",
                "description": "30 minutes of exercise",
                "frequency": "daily"
            }
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            details = f"Habit created: {data['habit']['title']}"
            return passed, data['habit']['id']
        else:
            details = f"Status: {response.status_code}"
            log_test("Create Habit", passed, details)
            return passed, None
        
    except Exception as e:
        log_test("Create Habit", False, str(e))
        return False, None


def test_complete_habit(habit_id: str):
    """Test habit completion"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/habits/{habit_id}/complete",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            details = f"Streak: {data['streak']}, XP: {data['xp_gained']}"
        else:
            details = f"Status: {response.status_code}"
        
        log_test("Complete Habit", passed, details)
        return passed
    except Exception as e:
        log_test("Complete Habit", False, str(e))
        return False


def test_journal_entry():
    """Test journal creation"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/journal",
            headers={"Authorization": f"Bearer {TOKEN}"},
            json={
                "content": "Today was great! Made amazing progress on my goals and feel wonderful.",
                "tags": ["productivity", "happiness"]
            }
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            details = f"Sentiment: {data['entry']['sentiment_score']:.2f}"
        else:
            details = f"Status: {response.status_code}"
        
        log_test("Journal Entry", passed, details)
        return passed
    except Exception as e:
        log_test("Journal Entry", False, str(e))
        return False


def test_pet_actions():
    """Test pet interactions"""
    try:
        # Feed pet
        response = requests.post(
            f"{BASE_URL}/api/pet/feed",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        passed1 = response.status_code == 200
        
        # Play with pet
        response = requests.post(
            f"{BASE_URL}/api/pet/play",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        passed2 = response.status_code == 200
        
        passed = passed1 and passed2
        details = "Feed & Play successful" if passed else "One or more actions failed"
        
        log_test("Pet Interactions", passed, details)
        return passed
    except Exception as e:
        log_test("Pet Interactions", False, str(e))
        return False


def test_fractal_generation():
    """Test fractal generation"""
    try:
        response = requests.get(
            f"{BASE_URL}/api/fractal/generate?type=auto",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        
        passed = response.status_code == 200 and response.headers.get('Content-Type') == 'image/png'
        
        if passed:
            size_kb = len(response.content) / 1024
            details = f"Generated {size_kb:.1f}KB PNG image"
        else:
            details = f"Status: {response.status_code}, Type: {response.headers.get('Content-Type')}"
        
        log_test("Fractal Generation", passed, details)
        return passed
    except Exception as e:
        log_test("Fractal Generation", False, str(e))
        return False


def test_fractal_metrics():
    """Test fractal metrics endpoint"""
    try:
        response = requests.get(
            f"{BASE_URL}/api/fractal/metrics",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            metrics = data['metrics']
            details = f"Momentum: {metrics['momentum']:.2f}, Type: {data['fractal_type']}"
        else:
            details = f"Status: {response.status_code}"
        
        log_test("Fractal Metrics", passed, details)
        return passed
    except Exception as e:
        log_test("Fractal Metrics", False, str(e))
        return False


def test_dashboard():
    """Test dashboard endpoint"""
    try:
        response = requests.get(
            f"{BASE_URL}/api/dashboard",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        
        passed = response.status_code == 200
        
        if passed:
            data = response.json()
            details = f"Goals: {len(data['recent_goals'])}, GPU: {data['gpu_status']['available']}"
        else:
            details = f"Status: {response.status_code}"
        
        log_test("Dashboard", passed, details)
        return passed
    except Exception as e:
        log_test("Dashboard", False, str(e))
        return False


def test_export_data():
    """Test data export"""
    try:
        response = requests.get(
            f"{BASE_URL}/api/export",
            headers={"Authorization": f"Bearer {TOKEN}"}
        )
        
        passed = response.status_code == 200 and 'application/json' in response.headers.get('Content-Type', '')
        
        if passed:
            size_kb = len(response.content) / 1024
            details = f"Exported {size_kb:.1f}KB backup"
        else:
            details = f"Status: {response.status_code}"
        
        log_test("Export Data", passed, details)
        return passed
    except Exception as e:
        log_test("Export Data", False, str(e))
        return False


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*80)
    print("🌀 LIFE FRACTAL INTELLIGENCE - API TEST SUITE")
    print("="*80 + "\n")
    
    print("📡 Testing Server Connection...")
    print("-" * 80)
    
    if not test_health_check():
        print("\n❌ Server not responding. Make sure it's running on http://localhost:5000")
        print("   Run: python life_fractal_complete.py")
        return
    
    print("\n👤 Testing Authentication...")
    print("-" * 80)
    test_register()
    test_login()
    
    print("\n🎯 Testing Goals & Tasks...")
    print("-" * 80)
    success, goal_id = test_create_goal()
    if success and goal_id:
        success, task_id = test_add_task(goal_id)
        if success and task_id:
            test_complete_task(task_id)
    
    print("\n📅 Testing Habits...")
    print("-" * 80)
    success, habit_id = test_create_habit()
    if success and habit_id:
        test_complete_habit(habit_id)
    
    print("\n📔 Testing Journal...")
    print("-" * 80)
    test_journal_entry()
    
    print("\n🐉 Testing Virtual Pet...")
    print("-" * 80)
    test_pet_actions()
    
    print("\n🎨 Testing Fractal Visualization...")
    print("-" * 80)
    test_fractal_metrics()
    test_fractal_generation()
    
    print("\n📊 Testing Dashboard & Data...")
    print("-" * 80)
    test_dashboard()
    test_export_data()
    
    print("\n" + "="*80)
    print("✅ TEST SUITE COMPLETE")
    print("="*80 + "\n")
    
    print("🎉 All core features tested!")
    print("📝 Check the output above for any failures.")
    print("\n💡 Next steps:")
    print("   1. Review any failed tests")
    print("   2. Test fractal generation in browser: http://localhost:5000")
    print("   3. Configure Stripe keys in .env for payment testing")
    print("   4. Deploy to production when ready!")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⏸️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
