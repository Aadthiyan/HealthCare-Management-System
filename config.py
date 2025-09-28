import os




# Supabase PostgreSQL connection using environment variables
# Set these in your .env file or system environment for security
SUPABASE_DB_HOST = os.getenv('SUPABASE_DB_HOST', 'aws-1-ap-south-1.pooler.supabase.com')
SUPABASE_DB_PORT = os.getenv('SUPABASE_DB_PORT', '5432')
SUPABASE_DB_NAME = os.getenv('SUPABASE_DB_NAME', 'postgres')
SUPABASE_DB_USER = os.getenv('SUPABASE_DB_USER', 'postgres.okywityiiyzmwymxecer')
SUPABASE_DB_PASSWORD = os.getenv('SUPABASE_DB_PASSWORD', 'aadhithiyan')  # Set your password securely

SQLALCHEMY_DATABASE_URI = (
    f"postgresql://{SUPABASE_DB_USER}:{SUPABASE_DB_PASSWORD}@{SUPABASE_DB_HOST}:{SUPABASE_DB_PORT}/{SUPABASE_DB_NAME}"
)
