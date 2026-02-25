from contextlib import contextmanager
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    Float,
    DateTime,
    text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import sys

# ============================================================================
# DATABASE CONFIG
# ============================================================================

# Database Connection
DATABASE_URL = "postgresql+psycopg2://infiniq_user:infiniq_pass@192.168.210.250:5432/word2map_db"

try:
    engine = create_engine(
        DATABASE_URL,
        echo=False,          # Set True to debug SQL
        future=True
    )
    # Test connection immediately
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        print(f"✅ Successfully connected to word2map_db!")
except Exception as e:
    print(f"❌ Failed to connect to database: {e}")
    sys.exit(1)

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False
)

Base = declarative_base()


@contextmanager
def get_session():
    """Safe session context manager"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ============================================================================
# BASE MODEL
# ============================================================================

class BaseModel(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# ============================================================================
# MODELS
# ============================================================================

class Floor(BaseModel):
    __tablename__ = "floors"

    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)

    cameras = relationship(
        "Camera",
        back_populates="floor",
        cascade="all, delete-orphan",
    )
    rois = relationship(
        "ROI",
        back_populates="floor",
        cascade="all, delete-orphan",
    )
    positions = relationship(
        "Position",
        back_populates="floor",
        cascade="all, delete-orphan",
    )


class Camera(BaseModel):
    __tablename__ = "cameras"

    ip = Column(String(50), unique=True, nullable=False)
    name = Column(String(100))
    floor_id = Column(Integer, ForeignKey("floors.id"))

    floor = relationship("Floor", back_populates="cameras")


class ROI(BaseModel):
    __tablename__ = "rois"

    name = Column(String(50), nullable=False)
    floor_id = Column(Integer, ForeignKey("floors.id"))

    x1 = Column(Integer, nullable=False)
    y1 = Column(Integer, nullable=False)
    x2 = Column(Integer, nullable=False)
    y2 = Column(Integer, nullable=False)

    floor = relationship("Floor", back_populates="rois")

    @property
    def width(self):
        return abs(self.x2 - self.x1)

    @property
    def height(self):
        return abs(self.y2 - self.y1)

    @property
    def area(self):
        return self.width * self.height


class Position(BaseModel):
    __tablename__ = "positions"

    name = Column(String(50), nullable=False)
    floor_id = Column(Integer, ForeignKey("floors.id"))

    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)

    floor = relationship("Floor", back_populates="positions")


# ============================================================================
# DATABASE HELPERS
# ============================================================================

def create_tables():
    print("Creating tables...")
    Base.metadata.create_all(engine)
    print("Tables created.")


def drop_tables():
    print("Dropping tables...")
    Base.metadata.drop_all(engine)
    print("Tables dropped.")


# ============================================================================
# SEED DATA
# ============================================================================

def seed_database():
    print("Seeding database...")
    try:
        with get_session() as session:
            # Check if exists
            if session.query(Floor).filter_by(name="Floor 1").first():
                print("Data already exists, skipping seed.")
                return

            floor = Floor(
                name="Floor 1",
                description="Main inspection floor",
            )

            floor.cameras = [
                Camera(ip="192.168.1.10", name="Camera 1"),
                Camera(ip="192.168.1.11", name="Camera 2"),
            ]

            floor.rois = [
                ROI(name="roi_1", x1=100, y1=200, x2=400, y2=600),
                ROI(name="roi_2", x1=500, y1=100, x2=900, y2=450),
            ]

            floor.positions = [
                Position(name="pos_A", x=12.5, y=30.8),
                Position(name="pos_B", x=50.0, y=18.2),
            ]

            session.add(floor)
            print("Database seeded successfully.")
    except Exception as e:
        print(f"Error seeding database: {e}")


# ============================================================================
# QUERY EXAMPLE
# ============================================================================

def get_floor_summary(floor_id: int):
    with get_session() as session:
        floor = session.get(Floor, floor_id)
        if not floor:
            return None

        return {
            "floor": floor.to_dict(),
            "cameras": [c.to_dict() for c in floor.cameras],
            "rois": [r.to_dict() for r in floor.rois],
            "positions": [p.to_dict() for p in floor.positions],
        }

if __name__ == "__main__":
    # Test flow
    try:
        create_tables()
        seed_database()
        
        # Verify data
        summary = get_floor_summary(1)
        if summary:
            print("\nVerification - Floor 1 Summary:")
            print(f"Name: {summary['floor']['name']}")
            print(f"Cameras: {len(summary['cameras'])}")
            print(f"ROIs: {len(summary['rois'])}")
        else:
            # It might be ID 1 doesn't exist if auto-icrement differs, try name lookup
            with get_session() as session:
                f = session.query(Floor).first()
                if f:
                     print(f"\nFound floor ID {f.id}: {f.name}")
    except Exception as e:
        print(f"\nAn error occurred during test: {e}")