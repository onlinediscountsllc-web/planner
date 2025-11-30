#!/usr/bin/env python3
"""
🎨 LIFE FRACTAL INTELLIGENCE - STUDIO GUI v5.0
═══════════════════════════════════════════════════════════════════════════════
Beautiful PySide6 desktop application that connects to the Flask API backend.
Features self-healing connections and graceful fallbacks.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import json
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps

# PySide6 imports
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QLineEdit, QTextEdit, QSlider, QComboBox,
        QTabWidget, QFrame, QScrollArea, QProgressBar, QMessageBox,
        QDialog, QFormLayout, QSpinBox, QCheckBox, QListWidget, QListWidgetItem,
        QStackedWidget, QGroupBox, QSplitter, QSizePolicy
    )
    from PySide6.QtCore import Qt, QTimer, Signal, QThread, QSize
    from PySide6.QtGui import QFont, QPixmap, QIcon, QPalette, QColor, QImage
    HAS_PYSIDE = True
except ImportError:
    HAS_PYSIDE = False
    print("❌ PySide6 not installed. Install with: pip install PySide6 --break-system-packages")
    sys.exit(1)

import requests
from io import BytesIO
import base64


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-HEALING API CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class SelfHealingAPIClient:
    """API client with automatic retry and fallback mechanisms."""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.max_retries = 3
        self.retry_delay = 1.0
        self.connected = False
        self.error_count = 0
        self.last_error = None
    
    def _retry_request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Make request with automatic retry."""
        url = f"{self.base_url}{endpoint}"
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                if method == 'GET':
                    response = self.session.get(url, timeout=10, **kwargs)
                elif method == 'POST':
                    response = self.session.post(url, timeout=10, **kwargs)
                elif method == 'PUT':
                    response = self.session.put(url, timeout=10, **kwargs)
                elif method == 'DELETE':
                    response = self.session.delete(url, timeout=10, **kwargs)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                response.raise_for_status()
                self.connected = True
                self.error_count = 0
                return response.json()
                
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                self.last_error = f"Connection failed: {e}"
                print(f"⚠️ Connection attempt {attempt + 1}/{self.max_retries} failed")
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                self.last_error = f"Request timed out: {e}"
                print(f"⚠️ Timeout on attempt {attempt + 1}/{self.max_retries}")
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                self.last_error = str(e)
                print(f"⚠️ Request failed: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        self.connected = False
        self.error_count += 1
        print(f"❌ All {self.max_retries} attempts failed for {endpoint}")
        return None
    
    def get(self, endpoint: str, **kwargs) -> Optional[Dict]:
        return self._retry_request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, data: Dict = None, **kwargs) -> Optional[Dict]:
        return self._retry_request('POST', endpoint, json=data, **kwargs)
    
    def put(self, endpoint: str, data: Dict = None, **kwargs) -> Optional[Dict]:
        return self._retry_request('PUT', endpoint, json=data, **kwargs)
    
    def check_health(self) -> bool:
        """Check if API is healthy."""
        result = self.get('/api/health')
        return result is not None and result.get('status') == 'healthy'


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC WORKER THREAD
# ═══════════════════════════════════════════════════════════════════════════════

class APIWorker(QThread):
    """Background worker for API calls."""
    
    finished = Signal(dict)
    error = Signal(str)
    
    def __init__(self, api_client: SelfHealingAPIClient, method: str, endpoint: str, data: Dict = None):
        super().__init__()
        self.api = api_client
        self.method = method
        self.endpoint = endpoint
        self.data = data
    
    def run(self):
        try:
            if self.method == 'GET':
                result = self.api.get(self.endpoint)
            elif self.method == 'POST':
                result = self.api.post(self.endpoint, self.data)
            else:
                result = None
            
            if result:
                self.finished.emit(result)
            else:
                self.error.emit(self.api.last_error or "Unknown error")
        except Exception as e:
            self.error.emit(str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# MODERN STYLED WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #0f172a;
    color: #f1f5f9;
    font-family: 'Segoe UI', system-ui, sans-serif;
}

QLabel {
    color: #f1f5f9;
}

QPushButton {
    background-color: #6366f1;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 8px;
    font-weight: bold;
    font-size: 13px;
}

QPushButton:hover {
    background-color: #4f46e5;
}

QPushButton:pressed {
    background-color: #3730a3;
}

QPushButton:disabled {
    background-color: #475569;
    color: #94a3b8;
}

QPushButton#success {
    background-color: #10b981;
}

QPushButton#success:hover {
    background-color: #059669;
}

QPushButton#golden {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #d4af37, stop:1 #b8860b);
}

QPushButton#outline {
    background-color: transparent;
    border: 2px solid #475569;
    color: #f1f5f9;
}

QPushButton#outline:hover {
    border-color: #6366f1;
    color: #6366f1;
}

QLineEdit, QTextEdit {
    background-color: #334155;
    border: 2px solid #475569;
    border-radius: 8px;
    padding: 10px;
    color: #f1f5f9;
    font-size: 14px;
}

QLineEdit:focus, QTextEdit:focus {
    border-color: #6366f1;
}

QSlider::groove:horizontal {
    border: none;
    height: 8px;
    background: #334155;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: #6366f1;
    width: 20px;
    height: 20px;
    margin: -6px 0;
    border-radius: 10px;
}

QSlider::handle:horizontal:hover {
    background: #4f46e5;
}

QTabWidget::pane {
    border: 1px solid #475569;
    border-radius: 8px;
    background: #1e293b;
}

QTabBar::tab {
    background: #334155;
    color: #94a3b8;
    padding: 12px 24px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #6366f1;
    color: white;
}

QTabBar::tab:hover:!selected {
    background: #475569;
    color: #f1f5f9;
}

QProgressBar {
    background-color: #334155;
    border-radius: 4px;
    height: 12px;
    text-align: center;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6366f1, stop:1 #10b981);
    border-radius: 4px;
}

QGroupBox {
    border: 1px solid #475569;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #d4af37;
}

QScrollArea {
    border: none;
    background: transparent;
}

QListWidget {
    background-color: #1e293b;
    border: 1px solid #475569;
    border-radius: 8px;
}

QListWidget::item {
    padding: 10px;
    border-bottom: 1px solid #334155;
}

QListWidget::item:selected {
    background-color: #6366f1;
}

QListWidget::item:hover:!selected {
    background-color: #334155;
}

QFrame#card {
    background-color: #1e293b;
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 16px;
}

QFrame#pet_card {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e293b, stop:1 #2d3748);
    border: 2px solid #d4af37;
    border-radius: 16px;
}

QLabel#title {
    font-size: 24px;
    font-weight: bold;
    color: #f1f5f9;
}

QLabel#subtitle {
    font-size: 14px;
    color: #94a3b8;
}

QLabel#stat_value {
    font-size: 36px;
    font-weight: bold;
    color: #6366f1;
}

QLabel#golden {
    color: #d4af37;
}

QLabel#success {
    color: #10b981;
}

QLabel#error {
    color: #ef4444;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class LifePlannerStudio(QMainWindow):
    """Main application window."""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        super().__init__()
        
        self.api = SelfHealingAPIClient(api_url)
        self.user_id = None
        self.user_data = None
        self.dashboard_data = None
        
        self.setWindowTitle("🌀 Life Fractal Intelligence - Studio")
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(DARK_STYLE)
        
        # Main container
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create pages
        self.login_page = self.create_login_page()
        self.dashboard_page = self.create_dashboard_page()
        
        self.central_widget.addWidget(self.login_page)
        self.central_widget.addWidget(self.dashboard_page)
        
        # Start on login page
        self.central_widget.setCurrentWidget(self.login_page)
        
        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_dashboard)
        
        # Check API health
        self.check_api_health()
    
    def check_api_health(self):
        """Check if API is available."""
        if self.api.check_health():
            self.status_label.setText("✅ API Connected")
            self.status_label.setObjectName("success")
        else:
            self.status_label.setText("❌ API Offline - Start the backend server")
            self.status_label.setObjectName("error")
        self.status_label.setStyleSheet(DARK_STYLE)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOGIN PAGE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_login_page(self) -> QWidget:
        """Create the login page."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        
        # Logo container
        logo_frame = QFrame()
        logo_layout = QVBoxLayout(logo_frame)
        logo_layout.setAlignment(Qt.AlignCenter)
        
        # Logo
        logo_label = QLabel("🌀")
        logo_label.setFont(QFont("Segoe UI", 72))
        logo_label.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(logo_label)
        
        # Title
        title = QLabel("Life Fractal Intelligence")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Sacred Geometry • Golden Ratio • Personal Growth")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(subtitle)
        
        layout.addWidget(logo_frame)
        layout.addSpacing(30)
        
        # Login form container
        form_frame = QFrame()
        form_frame.setFixedWidth(400)
        form_frame.setObjectName("card")
        form_layout = QVBoxLayout(form_frame)
        form_layout.setSpacing(15)
        
        # Status
        self.status_label = QLabel("Checking API connection...")
        self.status_label.setAlignment(Qt.AlignCenter)
        form_layout.addWidget(self.status_label)
        
        # Email
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")
        self.email_input.setText("onlinediscountsllc@gmail.com")
        form_layout.addWidget(self.email_input)
        
        # Password
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setText("admin8587037321")
        form_layout.addWidget(self.password_input)
        
        # Login button
        login_btn = QPushButton("🔐 Sign In")
        login_btn.clicked.connect(self.do_login)
        form_layout.addWidget(login_btn)
        
        # Demo login
        demo_btn = QPushButton("✨ Quick Demo Login")
        demo_btn.setObjectName("golden")
        demo_btn.clicked.connect(self.demo_login)
        form_layout.addWidget(demo_btn)
        
        # Register link
        register_btn = QPushButton("Create Account")
        register_btn.setObjectName("outline")
        register_btn.clicked.connect(self.show_register_dialog)
        form_layout.addWidget(register_btn)
        
        layout.addWidget(form_frame)
        
        return page
    
    def demo_login(self):
        """Quick demo login."""
        self.email_input.setText("onlinediscountsllc@gmail.com")
        self.password_input.setText("admin8587037321")
        self.do_login()
    
    def do_login(self):
        """Perform login."""
        email = self.email_input.text()
        password = self.password_input.text()
        
        if not email or not password:
            QMessageBox.warning(self, "Error", "Please enter email and password")
            return
        
        result = self.api.post('/api/auth/login', {'email': email, 'password': password})
        
        if result and not result.get('error'):
            self.user_id = result['user']['id']
            self.user_data = result['user']
            self.central_widget.setCurrentWidget(self.dashboard_page)
            self.load_dashboard()
            self.refresh_timer.start(30000)  # Refresh every 30 seconds
        else:
            error = result.get('error', 'Connection failed') if result else 'Connection failed'
            QMessageBox.warning(self, "Login Failed", error)
    
    def show_register_dialog(self):
        """Show registration dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Create Account")
        dialog.setFixedWidth(400)
        
        layout = QFormLayout(dialog)
        
        first_name = QLineEdit()
        last_name = QLineEdit()
        email = QLineEdit()
        password = QLineEdit()
        password.setEchoMode(QLineEdit.Password)
        
        layout.addRow("First Name:", first_name)
        layout.addRow("Last Name:", last_name)
        layout.addRow("Email:", email)
        layout.addRow("Password:", password)
        
        register_btn = QPushButton("🚀 Start Free Trial")
        register_btn.setObjectName("success")
        layout.addRow(register_btn)
        
        def do_register():
            result = self.api.post('/api/auth/register', {
                'email': email.text(),
                'password': password.text(),
                'first_name': first_name.text(),
                'last_name': last_name.text()
            })
            
            if result and not result.get('error'):
                QMessageBox.information(dialog, "Success", "Account created! 🎉")
                dialog.accept()
                self.user_id = result['user']['id']
                self.user_data = result['user']
                self.central_widget.setCurrentWidget(self.dashboard_page)
                self.load_dashboard()
            else:
                error = result.get('error', 'Registration failed') if result else 'Connection failed'
                QMessageBox.warning(dialog, "Error", error)
        
        register_btn.clicked.connect(do_register)
        dialog.exec()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DASHBOARD PAGE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_dashboard_page(self) -> QWidget:
        """Create the main dashboard page."""
        page = QWidget()
        main_layout = QVBoxLayout(page)
        main_layout.setSpacing(20)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Stats row
        stats_row = self.create_stats_row()
        main_layout.addWidget(stats_row)
        
        # Main content area with tabs
        tabs = QTabWidget()
        tabs.addTab(self.create_overview_tab(), "🏠 Overview")
        tabs.addTab(self.create_checkin_tab(), "📊 Daily Check-in")
        tabs.addTab(self.create_habits_tab(), "✅ Habits")
        tabs.addTab(self.create_goals_tab(), "🎯 Goals")
        tabs.addTab(self.create_pet_tab(), "🐾 Pet")
        tabs.addTab(self.create_fractal_tab(), "🌀 Fractal")
        
        main_layout.addWidget(tabs)
        
        return page
    
    def create_header(self) -> QFrame:
        """Create header bar."""
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #6366f1, stop:1 #8b5cf6);
                border-radius: 16px;
                padding: 20px;
            }
        """)
        
        layout = QHBoxLayout(header)
        
        # Logo and title
        logo_layout = QHBoxLayout()
        logo = QLabel("🌀")
        logo.setFont(QFont("Segoe UI", 36))
        logo_layout.addWidget(logo)
        
        title_layout = QVBoxLayout()
        title = QLabel("Life Fractal Intelligence")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: white;")
        subtitle = QLabel("Sacred Geometry • Golden Ratio • Personal Growth")
        subtitle.setStyleSheet("color: rgba(255,255,255,0.8);")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        logo_layout.addLayout(title_layout)
        
        layout.addLayout(logo_layout)
        layout.addStretch()
        
        # User info
        user_layout = QVBoxLayout()
        self.header_user_name = QLabel("Loading...")
        self.header_user_name.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.header_user_name.setStyleSheet("color: white;")
        self.header_user_status = QLabel("Checking status...")
        self.header_user_status.setStyleSheet("color: rgba(255,255,255,0.8);")
        user_layout.addWidget(self.header_user_name, alignment=Qt.AlignRight)
        user_layout.addWidget(self.header_user_status, alignment=Qt.AlignRight)
        layout.addLayout(user_layout)
        
        # Logout button
        logout_btn = QPushButton("Logout")
        logout_btn.setObjectName("outline")
        logout_btn.setStyleSheet("color: white; border-color: white;")
        logout_btn.clicked.connect(self.do_logout)
        layout.addWidget(logout_btn)
        
        return header
    
    def create_stats_row(self) -> QFrame:
        """Create stats row."""
        frame = QFrame()
        layout = QHBoxLayout(frame)
        layout.setSpacing(20)
        
        # Wellness
        self.wellness_card = self.create_stat_card("🧘", "Wellness Index", "--")
        layout.addWidget(self.wellness_card)
        
        # Streak
        self.streak_card = self.create_stat_card("🔥", "Day Streak", "--")
        layout.addWidget(self.streak_card)
        
        # Goals
        self.goals_stat_card = self.create_stat_card("🎯", "Goals Progress", "--%")
        layout.addWidget(self.goals_stat_card)
        
        # Habits
        self.habits_stat_card = self.create_stat_card("✨", "Habits Today", "--")
        layout.addWidget(self.habits_stat_card)
        
        return frame
    
    def create_stat_card(self, icon: str, label: str, value: str) -> QFrame:
        """Create a stat card widget."""
        card = QFrame()
        card.setObjectName("card")
        card.setMinimumHeight(120)
        
        layout = QVBoxLayout(card)
        layout.setAlignment(Qt.AlignCenter)
        
        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Segoe UI", 32))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        value_label = QLabel(value)
        value_label.setObjectName("stat_value")
        value_label.setAlignment(Qt.AlignCenter)
        card.value_label = value_label
        layout.addWidget(value_label)
        
        name_label = QLabel(label)
        name_label.setObjectName("subtitle")
        name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(name_label)
        
        return card
    
    def create_overview_tab(self) -> QWidget:
        """Create overview tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setSpacing(20)
        
        # Left column - Pet
        left = QFrame()
        left.setObjectName("pet_card")
        left_layout = QVBoxLayout(left)
        left_layout.setAlignment(Qt.AlignCenter)
        
        self.pet_emoji = QLabel("🐱")
        self.pet_emoji.setFont(QFont("Segoe UI", 80))
        self.pet_emoji.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.pet_emoji)
        
        self.pet_name_label = QLabel("Your Pet")
        self.pet_name_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.pet_name_label.setObjectName("golden")
        self.pet_name_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.pet_name_label)
        
        self.pet_behavior_label = QLabel("idle")
        self.pet_behavior_label.setObjectName("subtitle")
        self.pet_behavior_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.pet_behavior_label)
        
        # Pet actions
        actions_layout = QHBoxLayout()
        feed_btn = QPushButton("🍖 Feed")
        feed_btn.clicked.connect(self.feed_pet)
        play_btn = QPushButton("🎾 Play")
        play_btn.setObjectName("success")
        play_btn.clicked.connect(self.play_with_pet)
        actions_layout.addWidget(feed_btn)
        actions_layout.addWidget(play_btn)
        left_layout.addLayout(actions_layout)
        
        layout.addWidget(left, 1)
        
        # Right column - Guidance
        right = QFrame()
        right.setObjectName("card")
        right_layout = QVBoxLayout(right)
        
        guidance_title = QLabel("💬 AI Guidance")
        guidance_title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        right_layout.addWidget(guidance_title)
        
        self.guidance_message = QLabel("Loading your personalized guidance...")
        self.guidance_message.setWordWrap(True)
        self.guidance_message.setStyleSheet("""
            background-color: rgba(99, 102, 241, 0.2);
            border-left: 4px solid #6366f1;
            padding: 15px;
            border-radius: 8px;
        """)
        right_layout.addWidget(self.guidance_message)
        
        self.pet_message = QLabel("Your pet is here for you!")
        self.pet_message.setWordWrap(True)
        self.pet_message.setStyleSheet("""
            background-color: rgba(16, 185, 129, 0.2);
            border-left: 4px solid #10b981;
            padding: 15px;
            border-radius: 8px;
        """)
        right_layout.addWidget(self.pet_message)
        
        right_layout.addStretch()
        
        # Sacred math
        math_group = QGroupBox("✨ Sacred Mathematics")
        math_layout = QVBoxLayout(math_group)
        math_layout.addWidget(QLabel("φ (Golden Ratio): 1.618033988749895"))
        math_layout.addWidget(QLabel("Golden Angle: 137.5077640500°"))
        math_layout.addWidget(QLabel("Fibonacci: 1, 1, 2, 3, 5, 8, 13..."))
        right_layout.addWidget(math_group)
        
        layout.addWidget(right, 2)
        
        return tab
    
    def create_checkin_tab(self) -> QWidget:
        """Create daily check-in tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        
        # Date
        date_label = QLabel(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
        date_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        scroll_layout.addWidget(date_label)
        
        # Sliders
        self.mood_slider = self.create_slider_group("😊 Mood", 1, 10, 5)
        scroll_layout.addWidget(self.mood_slider)
        
        self.energy_slider = self.create_slider_group("⚡ Energy", 1, 10, 5)
        scroll_layout.addWidget(self.energy_slider)
        
        self.focus_slider = self.create_slider_group("🧠 Focus", 1, 10, 5)
        scroll_layout.addWidget(self.focus_slider)
        
        self.anxiety_slider = self.create_slider_group("😰 Anxiety", 1, 10, 3)
        scroll_layout.addWidget(self.anxiety_slider)
        
        self.sleep_slider = self.create_slider_group("💤 Sleep Hours", 0, 12, 7)
        scroll_layout.addWidget(self.sleep_slider)
        
        # Journal
        journal_group = QGroupBox("📝 Journal Entry")
        journal_layout = QVBoxLayout(journal_group)
        self.journal_text = QTextEdit()
        self.journal_text.setPlaceholderText("How was your day? What are you grateful for?")
        self.journal_text.setMinimumHeight(150)
        journal_layout.addWidget(self.journal_text)
        scroll_layout.addWidget(journal_group)
        
        # Save button
        save_btn = QPushButton("💾 Save Today's Entry")
        save_btn.clicked.connect(self.save_daily_entry)
        scroll_layout.addWidget(save_btn)
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        
        return tab
    
    def create_slider_group(self, label: str, min_val: int, max_val: int, default: int) -> QFrame:
        """Create a slider group."""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 10)
        
        header = QHBoxLayout()
        name_label = QLabel(label)
        name_label.setFont(QFont("Segoe UI", 12))
        value_label = QLabel(str(default))
        value_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        value_label.setObjectName("golden")
        header.addWidget(name_label)
        header.addStretch()
        header.addWidget(value_label)
        layout.addLayout(header)
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        layout.addWidget(slider)
        
        frame.slider = slider
        frame.value_label = value_label
        
        return frame
    
    def create_habits_tab(self) -> QWidget:
        """Create habits tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        header = QHBoxLayout()
        title = QLabel("✅ Your Habits")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header.addWidget(title)
        header.addStretch()
        
        add_btn = QPushButton("+ Add Habit")
        add_btn.clicked.connect(self.add_habit)
        header.addWidget(add_btn)
        layout.addLayout(header)
        
        self.habits_list = QListWidget()
        self.habits_list.setMinimumHeight(400)
        layout.addWidget(self.habits_list)
        
        return tab
    
    def create_goals_tab(self) -> QWidget:
        """Create goals tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        header = QHBoxLayout()
        title = QLabel("🎯 Your Goals")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        header.addWidget(title)
        header.addStretch()
        
        add_btn = QPushButton("+ Add Goal")
        add_btn.clicked.connect(self.add_goal)
        header.addWidget(add_btn)
        layout.addLayout(header)
        
        self.goals_list = QListWidget()
        self.goals_list.setMinimumHeight(400)
        layout.addWidget(self.goals_list)
        
        return tab
    
    def create_pet_tab(self) -> QWidget:
        """Create pet tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignCenter)
        
        # Large pet display
        pet_frame = QFrame()
        pet_frame.setObjectName("pet_card")
        pet_frame.setFixedSize(500, 500)
        pet_layout = QVBoxLayout(pet_frame)
        pet_layout.setAlignment(Qt.AlignCenter)
        
        self.big_pet_emoji = QLabel("🐱")
        self.big_pet_emoji.setFont(QFont("Segoe UI", 120))
        self.big_pet_emoji.setAlignment(Qt.AlignCenter)
        pet_layout.addWidget(self.big_pet_emoji)
        
        self.big_pet_name = QLabel("Loading...")
        self.big_pet_name.setFont(QFont("Segoe UI", 32, QFont.Bold))
        self.big_pet_name.setObjectName("golden")
        self.big_pet_name.setAlignment(Qt.AlignCenter)
        pet_layout.addWidget(self.big_pet_name)
        
        self.big_pet_level = QLabel("Level 1")
        self.big_pet_level.setFont(QFont("Segoe UI", 18))
        self.big_pet_level.setAlignment(Qt.AlignCenter)
        pet_layout.addWidget(self.big_pet_level)
        
        layout.addWidget(pet_frame)
        
        # Stats
        stats_frame = QFrame()
        stats_layout = QHBoxLayout(stats_frame)
        
        for stat in [("❤️ Hunger", "pet_hunger"), ("⚡ Energy", "pet_energy"), 
                     ("😊 Mood", "pet_mood"), ("💫 Bond", "pet_bond")]:
            stat_widget = QFrame()
            stat_widget.setObjectName("card")
            stat_inner = QVBoxLayout(stat_widget)
            stat_inner.setAlignment(Qt.AlignCenter)
            stat_inner.addWidget(QLabel(stat[0]))
            value = QLabel("50%")
            value.setObjectName("stat_value")
            setattr(self, stat[1], value)
            stat_inner.addWidget(value)
            stats_layout.addWidget(stat_widget)
        
        layout.addWidget(stats_frame)
        
        return tab
    
    def create_fractal_tab(self) -> QWidget:
        """Create fractal visualization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setAlignment(Qt.AlignCenter)
        
        title = QLabel("🌀 Your Personal Fractal")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Generated based on your mood and wellness data")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)
        
        self.fractal_label = QLabel("Loading fractal...")
        self.fractal_label.setAlignment(Qt.AlignCenter)
        self.fractal_label.setMinimumSize(512, 512)
        self.fractal_label.setStyleSheet("""
            border: 3px solid #d4af37;
            border-radius: 16px;
            background-color: #1e293b;
        """)
        layout.addWidget(self.fractal_label)
        
        refresh_btn = QPushButton("🔄 Refresh Fractal")
        refresh_btn.clicked.connect(self.refresh_fractal)
        layout.addWidget(refresh_btn)
        
        return tab
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA LOADING & ACTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def load_dashboard(self):
        """Load all dashboard data."""
        if not self.user_id:
            return
        
        result = self.api.get(f'/api/user/{self.user_id}/dashboard')
        if result:
            self.dashboard_data = result
            self.update_ui()
    
    def refresh_dashboard(self):
        """Refresh dashboard data."""
        self.load_dashboard()
    
    def update_ui(self):
        """Update all UI elements with loaded data."""
        if not self.dashboard_data:
            return
        
        data = self.dashboard_data
        
        # Header
        user = data.get('user', {})
        self.header_user_name.setText(f"{user.get('first_name', 'User')} {user.get('last_name', '')}")
        status = "✅ Active" if user.get('has_access') else f"Trial: {user.get('trial_days_remaining', 0)} days"
        self.header_user_status.setText(status)
        
        # Stats
        stats = data.get('stats', {})
        self.wellness_card.value_label.setText(str(int(stats.get('wellness_index', 0))))
        self.streak_card.value_label.setText(str(stats.get('current_streak', 0)))
        self.goals_stat_card.value_label.setText(f"{int(stats.get('goals_progress', 0))}%")
        habits_today = f"{stats.get('habits_completed_today', 0)}/{len(data.get('habits', []))}"
        self.habits_stat_card.value_label.setText(habits_today)
        
        # Pet
        pet = data.get('pet', {})
        if pet:
            emojis = {'cat': '🐱', 'dragon': '🐲', 'phoenix': '🔥', 'owl': '🦉', 'fox': '🦊'}
            emoji = emojis.get(pet.get('species', 'cat'), '🐱')
            self.pet_emoji.setText(emoji)
            self.big_pet_emoji.setText(emoji)
            self.pet_name_label.setText(pet.get('name', 'Pet'))
            self.big_pet_name.setText(pet.get('name', 'Pet'))
            self.pet_behavior_label.setText(pet.get('behavior', 'idle'))
            self.big_pet_level.setText(f"Level {pet.get('level', 1)}")
            
            self.pet_hunger.setText(f"{int(pet.get('hunger', 50))}%")
            self.pet_energy.setText(f"{int(pet.get('energy', 50))}%")
            self.pet_mood.setText(f"{int(pet.get('mood', 50))}%")
            self.pet_bond.setText(f"{int(pet.get('bond', 0))}%")
        
        # Today's entry
        today = data.get('today', {})
        if today:
            self.mood_slider.slider.setValue(int(today.get('mood_level', 3) * 2))
            self.energy_slider.slider.setValue(int(today.get('energy_level', 50) / 10))
            self.focus_slider.slider.setValue(int(today.get('focus_clarity', 50) / 10))
            self.anxiety_slider.slider.setValue(int(today.get('anxiety_level', 30) / 10))
            self.sleep_slider.slider.setValue(int(today.get('sleep_hours', 7)))
            self.journal_text.setText(today.get('journal_entry', ''))
        
        # Habits
        self.habits_list.clear()
        completed = today.get('habits_completed', {}) if today else {}
        for habit in data.get('habits', []):
            item = QListWidgetItem()
            check = "✅" if completed.get(habit['id']) else "⬜"
            item.setText(f"{check} {habit['name']} (🔥 {habit['current_streak']} day streak)")
            item.setData(Qt.UserRole, habit['id'])
            self.habits_list.addItem(item)
        
        # Goals
        self.goals_list.clear()
        for goal in data.get('goals', []):
            if not goal.get('is_completed'):
                item = QListWidgetItem()
                item.setText(f"🎯 {goal['title']} - {int(goal['progress'])}%")
                item.setData(Qt.UserRole, goal['id'])
                self.goals_list.addItem(item)
        
        # Guidance
        guidance = self.api.get(f'/api/user/{self.user_id}/guidance')
        if guidance:
            self.guidance_message.setText(guidance.get('fuzzy_message', 'Stay positive!'))
            self.pet_message.setText(guidance.get('pet_message', 'Your pet is happy!'))
        
        # Fractal
        self.refresh_fractal()
    
    def refresh_fractal(self):
        """Refresh the fractal image."""
        if not self.user_id:
            return
        
        result = self.api.get(f'/api/user/{self.user_id}/fractal/base64')
        if result and result.get('image'):
            # Decode base64 image
            image_data = result['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            pixmap = QPixmap()
            pixmap.loadFromData(image_bytes)
            scaled = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.fractal_label.setPixmap(scaled)
    
    def save_daily_entry(self):
        """Save daily check-in."""
        if not self.user_id:
            return
        
        data = {
            'mood_level': self.mood_slider.slider.value() // 2,
            'mood_score': self.mood_slider.slider.value() * 10,
            'energy_level': self.energy_slider.slider.value() * 10,
            'focus_clarity': self.focus_slider.slider.value() * 10,
            'anxiety_level': self.anxiety_slider.slider.value() * 10,
            'sleep_hours': float(self.sleep_slider.slider.value()),
            'journal_entry': self.journal_text.toPlainText()
        }
        
        result = self.api.post(f'/api/user/{self.user_id}/today', data)
        if result:
            QMessageBox.information(self, "Success", "Daily entry saved! ✅")
            self.load_dashboard()
        else:
            QMessageBox.warning(self, "Error", "Failed to save entry")
    
    def feed_pet(self):
        """Feed the pet."""
        if not self.user_id:
            return
        result = self.api.post(f'/api/user/{self.user_id}/pet/feed')
        if result:
            self.load_dashboard()
    
    def play_with_pet(self):
        """Play with the pet."""
        if not self.user_id:
            return
        result = self.api.post(f'/api/user/{self.user_id}/pet/play')
        if result and result.get('error'):
            QMessageBox.information(self, "Pet Tired", "Your pet is too tired to play! Let them rest. 😴")
        else:
            self.load_dashboard()
    
    def add_habit(self):
        """Add a new habit."""
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Add Habit", "Enter habit name:")
        if ok and name:
            self.api.post(f'/api/user/{self.user_id}/habits', {'name': name})
            self.load_dashboard()
    
    def add_goal(self):
        """Add a new goal."""
        from PySide6.QtWidgets import QInputDialog
        title, ok = QInputDialog.getText(self, "Add Goal", "Enter goal title:")
        if ok and title:
            self.api.post(f'/api/user/{self.user_id}/goals', {'title': title})
            self.load_dashboard()
    
    def do_logout(self):
        """Logout."""
        self.user_id = None
        self.user_data = None
        self.dashboard_data = None
        self.refresh_timer.stop()
        self.central_widget.setCurrentWidget(self.login_page)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("🎨 LIFE FRACTAL INTELLIGENCE - STUDIO GUI v5.0")
    print("=" * 60)
    print("Connecting to API at http://localhost:5000")
    print("Make sure the backend server is running!")
    print("=" * 60 + "\n")
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = LifePlannerStudio()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
