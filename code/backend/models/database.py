# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import os
import sys
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(parent_dir)
from constants import RDS_URL, RDS_PASSWORD

# DATABASE_URL = "sqlite:///./code/backend/test.db"  # easy local 

DATABASE_URL = f"postgresql+psycopg2://pathcoach_user:{RDS_PASSWORD}@{RDS_URL}:5432/pathcoach"

engine = create_engine(
    DATABASE_URL,
    echo=True,           # optional, for debug
    future=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
