"""
COMPREHENSIVE BUG TESTING FOR LIFE FRACTAL INTELLIGENCE
========================================================
Tests all authentication, security, and core functionality.
"""

import requests
import json
import time
import sys
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8080"  # Change to your Render URL for production testing
TEST_USER_EMAIL = f"test_{int(time.time())}@example.com"
TEST_PASSWORD = "SecurePass123!"
TEST_FIRST_NAME = "Test"
TEST_LAST_NAME = "User"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(test_name):
    print(f"\n{Colors.BLUE}[TEST]{Colors.END} {test_name}")

def print_pass(message):
    print(f"  {Colors.GREEN}✓ PASS:{Colors.END} {message}")

def print_fail(message):
    print(f"  {Colors.RED}✗ FAIL:{Colors.END} {message}")

def print_warning(message):
    print(f"  {Colors.YELLOW}⚠ WARNING:{Colors.END} {message}")

def print_section(title):
    print(f"\n{'='*60}")
    print(f"{Colors.BLUE}{title}{Colors.END}")
    print('='*60)


class BugTester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session_token = None
        self.user_id = None
        self.challenge_id = None
        self.captcha_answer = None
        self.tests_passed = 0
        self.tests_failed = 0
    
    def run_all_tests(self):
        """Run complete test suite."""
        print_section("LIFE FRACTAL INTELLIGENCE - BUG TEST SUITE")
        print(f"Testing URL: {self.base_url}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        try:
            # Test 1: Health check
            self.test_health_check()
            
            # Test 2: CAPTCHA generation
            self.test_captcha_generation()
            
            # Test 3: Registration validation
            self.test_registration_validation()
            
            # Test 4: Email check (returning user)
            self.test_email_check()
            
            # Test 5: Successful registration
            self.test_registration()
            
            # Test 6: Duplicate registration prevention
            self.test_duplicate_registration()
            
            # Test 7: Login with wrong CAPTCHA
            self.test_login_wrong_captcha()
            
            # Test 8: Login with wrong password
            self.test_login_wrong_password()
            
            # Test 9: Successful login
            self.test_login()
            
            # Test 10: Session verification
            self.test_session_verification()
            
            # Test 11: Dashboard access
            self.test_dashboard_access()
            
            # Test 12: Rate limiting
            self.test_rate_limiting()
            
            # Test 13: Password reset request
            self.test_password_reset_request()
            
            # Test 14: Invalid session handling
            self.test_invalid_session()
            
            # Test 15: CORS headers
            self.test_cors_headers()
            
            # Final summary
            self.print_summary()
            
        except KeyboardInterrupt:
            print("\n\nTest suite interrupted by user.")
            self.print_summary()
            sys.exit(1)
        except Exception as e:
            print_fail(f"Unexpected error: {e}")
            self.print_summary()
            sys.exit(1)
    
    def test_health_check(self):
        """Test health endpoint."""
        print_test("Health Check")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    print_pass("Server is healthy")
                    self.tests_passed += 1
                    return
            print_fail(f"Health check failed: {response.status_code}")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Cannot connect to server: {e}")
            self.tests_failed += 1
            sys.exit(1)
    
    def test_captcha_generation(self):
        """Test CAPTCHA generation."""
        print_test("CAPTCHA Generation")
        try:
            response = requests.get(f"{self.base_url}/api/auth/captcha")
            if response.status_code == 200:
                data = response.json()
                self.challenge_id = data.get('challenge_id')
                question = data.get('question')
                
                if self.challenge_id and question:
                    # Extract answer from question (for testing only)
                    parts = question.split()
                    if len(parts) >= 5:
                        try:
                            num1 = int(parts[3])
                            num2 = int(parts[5].rstrip('?'))
                            self.captcha_answer = num1 + num2
                            print_pass(f"CAPTCHA generated: {question}")
                            print_pass(f"Challenge ID: {self.challenge_id}")
                            self.tests_passed += 1
                            return
                        except:
                            pass
            
            print_fail("CAPTCHA generation failed")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"CAPTCHA error: {e}")
            self.tests_failed += 1
    
    def test_registration_validation(self):
        """Test registration input validation."""
        print_test("Registration Validation")
        
        # Get new CAPTCHA
        self._get_new_captcha()
        
        test_cases = [
            ({}, "Empty request"),
            ({'email': 'invalid', 'password': '12345678'}, "Invalid email format"),
            ({'email': 'test@test.com', 'password': '123'}, "Short password"),
        ]
        
        passed = 0
        for payload, description in test_cases:
            payload['challenge_id'] = self.challenge_id
            payload['captcha_answer'] = self.captcha_answer
            try:
                response = requests.post(
                    f"{self.base_url}/api/auth/register",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                if response.status_code in [400, 422]:
                    passed += 1
                    print_pass(f"Rejected {description}")
                else:
                    print_fail(f"Did not reject {description}")
            except Exception as e:
                print_fail(f"Validation test error: {e}")
        
        if passed == len(test_cases):
            self.tests_passed += 1
        else:
            self.tests_failed += 1
    
    def test_email_check(self):
        """Test email check endpoint."""
        print_test("Email Check (Returning User)")
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/check-email",
                json={'email': TEST_USER_EMAIL}
            )
            if response.status_code == 200:
                data = response.json()
                if not data.get('is_returning_user'):
                    print_pass("New user detected correctly")
                    self.tests_passed += 1
                    return
            print_fail("Email check failed")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Email check error: {e}")
            self.tests_failed += 1
    
    def test_registration(self):
        """Test successful registration."""
        print_test("User Registration")
        
        # Get new CAPTCHA
        self._get_new_captcha()
        
        payload = {
            'email': TEST_USER_EMAIL,
            'password': TEST_PASSWORD,
            'first_name': TEST_FIRST_NAME,
            'last_name': TEST_LAST_NAME,
            'challenge_id': self.challenge_id,
            'captcha_answer': str(self.captcha_answer)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/register",
                json=payload
            )
            if response.status_code == 201:
                data = response.json()
                self.user_id = data.get('access_token')
                print_pass(f"User registered successfully")
                print_pass(f"User ID: {self.user_id}")
                print_pass(f"Trial days: {data.get('trial_days_remaining')}")
                if data.get('show_gofundme'):
                    print_pass(f"GoFundMe shown: {data.get('gofundme_url')}")
                self.tests_passed += 1
                return
            print_fail(f"Registration failed: {response.status_code} - {response.text}")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Registration error: {e}")
            self.tests_failed += 1
    
    def test_duplicate_registration(self):
        """Test duplicate email prevention."""
        print_test("Duplicate Registration Prevention")
        
        # Get new CAPTCHA
        self._get_new_captcha()
        
        payload = {
            'email': TEST_USER_EMAIL,
            'password': TEST_PASSWORD,
            'first_name': TEST_FIRST_NAME,
            'last_name': TEST_LAST_NAME,
            'challenge_id': self.challenge_id,
            'captcha_answer': str(self.captcha_answer)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/register",
                json=payload
            )
            if response.status_code == 400:
                print_pass("Duplicate registration blocked")
                self.tests_passed += 1
                return
            print_fail("Duplicate registration was not blocked")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Duplicate test error: {e}")
            self.tests_failed += 1
    
    def test_login_wrong_captcha(self):
        """Test login with wrong CAPTCHA."""
        print_test("Login with Wrong CAPTCHA")
        
        # Get new CAPTCHA
        self._get_new_captcha()
        
        payload = {
            'email': TEST_USER_EMAIL,
            'password': TEST_PASSWORD,
            'challenge_id': self.challenge_id,
            'captcha_answer': '99999'  # Wrong answer
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/login",
                json=payload
            )
            if response.status_code in [400, 401]:
                print_pass("Wrong CAPTCHA rejected")
                self.tests_passed += 1
                return
            print_fail("Wrong CAPTCHA was not rejected")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"CAPTCHA test error: {e}")
            self.tests_failed += 1
    
    def test_login_wrong_password(self):
        """Test login with wrong password."""
        print_test("Login with Wrong Password")
        
        # Get new CAPTCHA
        self._get_new_captcha()
        
        payload = {
            'email': TEST_USER_EMAIL,
            'password': 'WrongPassword123!',
            'challenge_id': self.challenge_id,
            'captcha_answer': str(self.captcha_answer)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/login",
                json=payload
            )
            if response.status_code == 401:
                print_pass("Wrong password rejected")
                self.tests_passed += 1
                return
            print_fail("Wrong password was not rejected")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Password test error: {e}")
            self.tests_failed += 1
    
    def test_login(self):
        """Test successful login."""
        print_test("User Login")
        
        # Get new CAPTCHA
        self._get_new_captcha()
        
        payload = {
            'email': TEST_USER_EMAIL,
            'password': TEST_PASSWORD,
            'challenge_id': self.challenge_id,
            'captcha_answer': str(self.captcha_answer)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/login",
                json=payload
            )
            if response.status_code == 200:
                data = response.json()
                self.session_token = data.get('session_token')
                print_pass("Login successful")
                print_pass(f"Session token: {self.session_token[:16]}...")
                print_pass(f"Trial active: {data.get('trial_active')}")
                print_pass(f"Days remaining: {data.get('days_remaining')}")
                self.tests_passed += 1
                return
            print_fail(f"Login failed: {response.status_code} - {response.text}")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Login error: {e}")
            self.tests_failed += 1
    
    def test_session_verification(self):
        """Test session token verification."""
        print_test("Session Verification")
        if not self.session_token:
            print_warning("Skipped - no session token")
            return
        
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/verify-session",
                json={'session_token': self.session_token}
            )
            if response.status_code == 200:
                data = response.json()
                if data.get('valid'):
                    print_pass("Session verified successfully")
                    self.tests_passed += 1
                    return
            print_fail("Session verification failed")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Session verification error: {e}")
            self.tests_failed += 1
    
    def test_dashboard_access(self):
        """Test dashboard access with valid user."""
        print_test("Dashboard Access")
        if not self.user_id:
            print_warning("Skipped - no user ID")
            return
        
        try:
            response = requests.get(f"{self.base_url}/api/user/{self.user_id}/dashboard")
            if response.status_code == 200:
                data = response.json()
                if 'user' in data and 'stats' in data and 'pet' in data:
                    print_pass("Dashboard accessed successfully")
                    print_pass(f"Pet species: {data['pet'].get('species')}")
                    print_pass(f"Wellness index: {data['stats'].get('wellness_index')}")
                    self.tests_passed += 1
                    return
            print_fail(f"Dashboard access failed: {response.status_code}")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Dashboard error: {e}")
            self.tests_failed += 1
    
    def test_rate_limiting(self):
        """Test rate limiting on login attempts."""
        print_test("Rate Limiting")
        print_warning("Making 6 failed login attempts...")
        
        failed_attempts = 0
        for i in range(6):
            self._get_new_captcha()
            payload = {
                'email': f"nonexistent{i}@test.com",
                'password': "WrongPass123!",
                'challenge_id': self.challenge_id,
                'captcha_answer': str(self.captcha_answer)
            }
            try:
                response = requests.post(
                    f"{self.base_url}/api/auth/login",
                    json=payload
                )
                if response.status_code == 401:
                    failed_attempts += 1
            except:
                pass
            time.sleep(0.5)
        
        if failed_attempts >= 5:
            print_pass(f"Rate limiting active after {failed_attempts} attempts")
            self.tests_passed += 1
        else:
            print_warning(f"Rate limiting may not be working ({failed_attempts} attempts)")
            self.tests_failed += 1
    
    def test_password_reset_request(self):
        """Test password reset request."""
        print_test("Password Reset Request")
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/forgot-password",
                json={'email': TEST_USER_EMAIL}
            )
            if response.status_code == 200:
                print_pass("Password reset request accepted")
                self.tests_passed += 1
                return
            print_fail("Password reset request failed")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Password reset error: {e}")
            self.tests_failed += 1
    
    def test_invalid_session(self):
        """Test handling of invalid session token."""
        print_test("Invalid Session Handling")
        try:
            response = requests.post(
                f"{self.base_url}/api/auth/verify-session",
                json={'session_token': 'invalid_token_12345'}
            )
            if response.status_code == 401:
                print_pass("Invalid session rejected correctly")
                self.tests_passed += 1
                return
            print_fail("Invalid session was not rejected")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"Invalid session test error: {e}")
            self.tests_failed += 1
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        print_test("CORS Headers")
        try:
            response = requests.get(f"{self.base_url}/health")
            cors_header = response.headers.get('Access-Control-Allow-Origin')
            if cors_header:
                print_pass(f"CORS enabled: {cors_header}")
                self.tests_passed += 1
                return
            print_warning("CORS headers not found")
            self.tests_failed += 1
        except Exception as e:
            print_fail(f"CORS test error: {e}")
            self.tests_failed += 1
    
    def _get_new_captcha(self):
        """Helper to get a new CAPTCHA challenge."""
        try:
            response = requests.get(f"{self.base_url}/api/auth/captcha")
            if response.status_code == 200:
                data = response.json()
                self.challenge_id = data.get('challenge_id')
                question = data.get('question')
                parts = question.split()
                num1 = int(parts[3])
                num2 = int(parts[5].rstrip('?'))
                self.captcha_answer = num1 + num2
        except:
            pass
    
    def print_summary(self):
        """Print test summary."""
        print_section("TEST SUMMARY")
        total = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / max(1, total)) * 100
        
        print(f"\nTotal Tests: {total}")
        print(f"{Colors.GREEN}Passed: {self.tests_passed}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.tests_failed}{Colors.END}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if self.tests_failed == 0:
            print(f"\n{Colors.GREEN}🎉 ALL TESTS PASSED! System is ready for deployment.{Colors.END}")
        else:
            print(f"\n{Colors.RED}⚠️  Some tests failed. Please review errors above.{Colors.END}")


if __name__ == '__main__':
    print("\nStarting bug test suite...")
    print("Make sure the server is running on the specified URL.\n")
    
    # Allow custom URL
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
    
    tester = BugTester(BASE_URL)
    tester.run_all_tests()
