from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta

# Creăm un fișier local numit salon.db
SQLALCHEMY_DATABASE_URL = "sqlite:///./salon.db"

# connect_args={"check_same_thread": False} este necesar doar pentru SQLite în FastAPI
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Definim Tabelul
class CallLog(Base):
    __tablename__ = "call_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True) # Ex: Numărul de telefon
    messages = Column(Text)                 # Istoricul conversației (JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Funcție pentru crearea bazei de date (dacă nu există)
def init_db():
    Base.metadata.create_all(bind=engine)