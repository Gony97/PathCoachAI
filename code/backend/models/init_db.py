from database import Base, engine
import models  # VERY important: this loads the model classes

print("Creating tables...")
Base.metadata.create_all(bind=engine)
print("Done.")