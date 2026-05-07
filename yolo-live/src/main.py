import uvicorn
# Ndryshoje importin në absolut. 
# Supozojmë se app.py është në të njëjtën folder me main.py
try:
    from app import create_app
except ImportError:
    # Nëse main.py është jashtë folderit të app, përdor emrin e folderit (p.sh. from src.app)
    from .app import create_app 

app = create_app()

if __name__ == "__main__":
    # Përdor string "main:app" në vend të objektit app për të lejuar 'reload' gjatë zhvillimit
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)