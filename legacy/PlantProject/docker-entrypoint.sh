#!/bin/bash
set -e

# Create required directories
mkdir -p /app/instance
mkdir -p /app/static/uploads
chmod 777 /app/static/uploads

# Create database tables first
echo "Initializing database..."
python -c "
from application import app, db
from application import User, History

with app.app_context():
    # Force create all database tables
    db.create_all()
    
    # Check if User table exists and has any users
    user_count = User.query.count()
    print(f'Database initialized with {user_count} existing users')
    print('Database tables created successfully')
"

echo "Application directories and database prepared successfully"

# Then run the application
echo "Starting application..."
exec "$@"
