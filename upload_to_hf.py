import os
from huggingface_hub import HfApi, create_repo

# ==========================================
# CONFIGURACIÓN - ¡Cámbialo con tus datos!
# ==========================================
HF_USERNAME = "Marianinpb"  # Ejemplo: "mtico"
TOOLS_REPO_NAME = "laya-toolrag-router"     # Nombre que tendrá en HF
CONTEXT_REPO_NAME = "laya-context-router"   # Nombre que tendrá en HF

# Rutas locales
TOOLS_FOLDER = r"drive-download-20260927T072804Z-1-001\laya-toolrag-router"
CONTEXT_FOLDER = r"drive-download-20260927T072804Z-1-002\laya-context-filter"

def upload_model(folder_path, repo_id):
    api = HfApi()
    
    print(f"\nCreando repositorio '{repo_id}' (si no existe)...")
    try:
        create_repo(repo_id, exist_ok=True, private=False) # Pon private=True si no quieres que sea público
    except Exception as e:
        print(f"Nota: {e}")
        
    print(f"Subiendo archivos de {folder_path} a HuggingFace Hub en {repo_id}...")
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model"
        )
        print("¡Subida completada con éxito! 🎉")
    except Exception as e:
        print(f"Error al subir: {e}")

if __name__ == "__main__":
    if HF_USERNAME == "tu_usuario_de_huggingface":
        print("⚠️ Por favor, edita este archivo y pon tu usuario de HuggingFace en HF_USERNAME.")
    else:
        print("Iniciando subida de modelos a HuggingFace...")
        upload_model(TOOLS_FOLDER, f"{HF_USERNAME}/{TOOLS_REPO_NAME}")
        upload_model(CONTEXT_FOLDER, f"{HF_USERNAME}/{CONTEXT_REPO_NAME}")
        
        print("\n" + "="*50)
        print("¡Todo listo! Para usar los modelos en tu código o en otra PC, solo pon:")
        print(f"router = Router(models={{'english': '{HF_USERNAME}/{TOOLS_REPO_NAME}'}})")
        print("="*50)
