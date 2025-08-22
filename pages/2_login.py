import streamlit as st
import hashlib
import json
import os
from datetime import datetime
import re

# Initialize session state for authentication
def init_auth_session_state():
    """Initialize authentication related session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'show_signup' not in st.session_state:
        st.session_state.show_signup = False
    if 'new_user' not in st.session_state:
        st.session_state.new_user = False

# File-based user storage (you can replace with database)
USER_DATA_FILE = "users_data.json"

def save_users_data(users_data):
    """Save users data to file"""
    newUserId = max(users_data['userId'])+1
    {'userId':newUserId}
    with open(USER_DATA_FILE, 'w') as f:
        json.dump(users_data, f, indent=2)

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_user_id(user_id):
    """Validate user ID format"""
    # User ID should be alphanumeric, allowing underscore and hyphen, 3-20 characters
    pattern = r'^[a-zA-Z0-9_-]{3,20}$'
    return re.match(pattern, user_id) is not None

def validate_password(password):
    """Validate password strength"""
    # At least 6 characters
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, "Valid password"

def create_new_user(user_id, display_name=""):
    """Create a new user account"""
    users_data = st.session_state.ratings_df_cache
    
    # Check if user already exists
    if len(users_data[users_data['userId']==user_id])!=0:
        return False, "User ID already exists"

    save_users_data(users_data)
    return True, "Account created successfully"

def authenticate_user(user_id):
    """Authenticate user login"""
    # users_data = load_users_data()
    
    # if user_id not in users_data:
    #     return False, "User ID not found"
    
    # return True, users_data[user_id]
    users_data = st.session_state.ratings_df_cache
    if len(users_data[users_data['userId']==user_id])!=0:
        st.session_state.new_user=True
    else:
        st.session_state.new_user=False
    return True

def login_page():
    """Display login/signup page"""
    init_auth_session_state()
    
    # Custom CSS for login page
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: auto;
            padding: 20px;
        }
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .switch-mode {
            text-align: center;
            margin-top: 20px;
            color: #888;
        }
        .stButton > button {
            width: 100%;
            margin-top: 10px;
        }
        .success-message {
            padding: 10px;
            border-radius: 5px;
            background-color: #d4edda;
            color: #155724;
            margin: 10px 0;
        }
        .error-message {
            padding: 10px;
            border-radius: 5px;
            background-color: #f8d7da;
            color: #721c24;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        
        # Toggle between login and signup
        # LOGIN FORM
        st.markdown("<p style='text-align: center; color: #888;'>Login to your account</p>", unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            user_id = st.text_input(
                "User ID",
                placeholder="Enter your User ID",
                help="Your unique user identifier"
            )
            
            
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
            
            if submitted:
                if not user_id:
                    st.error("Please fill in all fields")
                else:
                    success = authenticate_user(user_id)
                    if success:
                        st.session_state.user_id = user_id
                        st.balloons()

# Logout function
def logout():
    """Logout current user"""
    st.session_state.authenticated = False
    st.session_state.user_id = None
    st.session_state.user_data = {}
    st.session_state.show_signup = False
    st.rerun()

# User profile widget
def show_user_profile():
    """Display user profile widget in sidebar"""
    if st.session_state.authenticated:
        with st.sidebar:
            st.divider()
            st.markdown("### 👤 User Profile")
            st.write(f"**User:** {st.session_state.user_data.get('display_name', st.session_state.user_id)}")
            st.write(f"**ID:** {st.session_state.user_id}")
            
            # Show user stats
            ratings_count = len(st.session_state.user_data.get('ratings', {}))
            st.write(f"**Movies Rated:** {ratings_count}")
            
            member_since = st.session_state.user_data.get('created_at', '')
            if member_since:
                date = datetime.fromisoformat(member_since).strftime("%B %Y")
                st.write(f"**Member Since:** {date}")
            
            if st.button("🚪 Logout", use_container_width=True):
                logout()

# Main app integration
def main_with_auth():
    """Main function that integrates authentication with the app"""
    init_auth_session_state()
    login_page()

# To use this in your main app:
if __name__ == "__main__":
    main_with_auth()