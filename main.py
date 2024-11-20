from brake_system import BrakeDiagnostic, BrakeProblem
from experta import Fact
from fastapi import FastAPI, HTTPException, Depends
from fastapi import Body
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
from typing import Dict, Optional, List, Any
import json
import os
import secrets
import pgmpy
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="Car Expert System API")

# Configuración de la base de datos
SQLALCHEMY_DATABASE_URL = os.getenv("DB_URL")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Configuración de JWT
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configuración de password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class QuestionResponse(BaseModel):
    answer: str

class DiagnosticResult(BaseModel):
    most_probable_problem: str
    probabilities: Dict[str, float]
    diagnostic_message: str

class DiagnosticSession:
    def __init__(self):
        self.engine = None
        self.evidence_list = []
        self.completed = False
        self.conversation = []  # Lista de diccionarios {"question": ..., "answer": ...}

sessions: Dict[str, DiagnosticSession] = {}

# Modelos de la base de datos
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    phone = Column(String)
    hashed_password = Column(String)

# Modelos Pydantic
class UserBase(BaseModel):
    email: EmailStr
    name: str
    phone: str

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class DiagnosticSessionRecord(Base):
    __tablename__ = "diagnosticsessions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    conversation = Column(String)  # Almacena la conversación como un JSON serializado
    diagnostic_result = Column(String)  # Almacena el diagnóstico final

# Funciones de utilidad
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# Rutas de la API
@app.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        name=user.name,
        phone=user.phone,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email, "name": user.name, "phone": user.phone}, 
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/api/diagnostic/start")
async def start_diagnostic(current_user: User = Depends(get_current_user)):
    """Inicia una nueva sesión de diagnóstico"""
    session_id = str(len(sessions) + 1)
    session = DiagnosticSession()
    engine = BrakeDiagnostic()
    engine.reset()
    engine.declare(Fact(action="diagnose_brakes"))
    engine.run()  # Esto activará la primera regla
    
    session.engine = engine
    sessions[session_id] = session
    
    return {
        "session_id": session_id,
        "question": engine.get_next_question()
    }

@app.post("/api/diagnostic/{session_id}/answer")
async def submit_answer(session_id: str, response: QuestionResponse, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = sessions[session_id]
    answer = response.answer.lower()
    
    if answer not in ["yes", "no"]:
        raise HTTPException(status_code=400, detail="Answer must be 'yes' or 'no'")

    # Almacenar la pregunta actual y la respuesta
    current_question = session.engine.get_next_question()
    session.conversation.append({"question": current_question, "answer": answer})

    # Procesar la respuesta actual
    session.engine.process_answer(answer)
    session.engine.run()

    # Obtener la siguiente pregunta
    next_question = session.engine.get_next_question()

    if next_question:
        return {
            "session_id": session_id,
            "question": next_question
        }
    else:
        # Generar el diagnóstico final
        diagnostic = session.engine.generate_diagnostic(dict(session.engine.evidence_list))
        sessions.pop(session_id)  # Limpiar la sesión

        # Guardar la conversación y el diagnóstico en la base de datos
        session_record = DiagnosticSessionRecord(
            user_id=current_user.id,
            conversation=json.dumps(session.conversation),
            diagnostic_result=json.dumps(diagnostic)
        )
        db.add(session_record)
        db.commit()

        return {
            "session_id": session_id,
            "diagnostic_result": diagnostic
        }


@app.get("/api/diagnostic/{session_id}")
async def get_diagnostic_status(session_id: str, current_user: User = Depends(get_current_user)):
    """Obtiene el estado actual del diagnóstico"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "current_question": session.engine.get_next_question(),
        "completed": session.engine.diagnostic_complete
    }

@app.get("/api/diagnostic/sessions", response_model=List[Dict[str, Any]])
async def get_user_diagnostics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(DiagnosticSessionRecord).filter(DiagnosticSessionRecord.user_id == current_user.id).all()
    return [
        {
            "id": session.id,
            "conversation": json.loads(session.conversation),
            "diagnostic_result": json.loads(session.diagnostic_result)
        }
        for session in sessions
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)