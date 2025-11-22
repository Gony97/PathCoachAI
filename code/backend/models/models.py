from sqlalchemy import Column, Integer, String, JSON, DateTime, Enum, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime
import enum


# ---------- ENUMS ----------

class UserType(enum.Enum):
    admin = "Admin"
    user = "User"


class EducationLevel(enum.Enum):
    undergrad = "Undergrad"
    bsc = "BSc"
    master = "Master"
    phd = "PhD"


class ModuleDifficulty(enum.Enum):
    easy = "Easy"
    medium = "Medium"
    hard = "Hard"


class PlanStatus(enum.Enum):
    active = "Active"
    completed = "Completed"
    paused = "Paused"
    cancelled = "Cancelled"


# ---------- MODELS ----------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    role = Column(Enum(UserType), nullable=False, default=UserType.user)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    profile = relationship("UserProfile", back_populates="user", uselist=False)
    plans = relationship("Plan", back_populates="user")


class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    education_level = Column(Enum(EducationLevel), nullable=True)
    years_experience = Column(Integer)
    dream_role = Column(String)
    interests = Column(JSON)        # e.g. ["AI", "Data Science"]
    relevant_roles = Column(JSON)   # e.g. ["ML Engineer", "Data Scientist"]
    knowledge_rating = Column(JSON) # e.g. {"python": 4, "sql": 3, "git": 2}

    user = relationship("User", back_populates="profile")


class Module(Base):
    __tablename__ = "modules"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    description = Column(String)
    skills_required = Column(JSON)      # e.g. ["python", "sql"]
    estimated_time_hours = Column(Integer)
    resources = Column(JSON)            # e.g. list of links
    difficulty = Column(Enum(ModuleDifficulty), nullable=False)
    learning_style = Column(JSON)       # e.g. ["reading", "hands-on"]
    outcome = Column(String)


class Plan(Base):
    __tablename__ = "plans"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    name = Column(String, nullable=False)
    description = Column(String)
    start = Column(DateTime, default=datetime.utcnow, nullable=False)
    end = Column(DateTime, nullable=True)
    status = Column(Enum(PlanStatus), nullable=False, default=PlanStatus.active)

    user = relationship("User", back_populates="plans")
    items = relationship("PlanItem", back_populates="plan", cascade="all, delete-orphan")


class PlanItem(Base):
    __tablename__ = "plan_items"

    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey("plans.id"), nullable=False)
    module_id = Column(Integer, ForeignKey("modules.id"), nullable=False)

    plan = relationship("Plan", back_populates="items")
    module = relationship("Module")
    progress_items = relationship("ProgressItem", back_populates="plan_item")


class ProgressItem(Base):
    __tablename__ = "progress_items"

    id = Column(Integer, primary_key=True)
    plan_item_id = Column(Integer, ForeignKey("plan_items.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    status = Column(Enum(PlanStatus), nullable=False, default=PlanStatus.active)
    start = Column(DateTime, nullable=True)
    end = Column(DateTime, nullable=True)

    plan_item = relationship("PlanItem", back_populates="progress_items")
    user = relationship("User")
