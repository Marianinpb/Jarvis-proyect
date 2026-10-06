import urllib.request
import json

OLLAMA_URL = "http://localhost:11434/api/generate"

# Reemplaza estos nombres con los nombres que le diste a tus modelos en Ollama
LAYA_TOOLS_MODEL = "laya-tools-finetuned:latest" 
LAYA_CONTEXT_MODEL = "laya-context-finetuned:latest"

def query_ollama(model, prompt):
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "")
    except Exception as e:
        return f"Error: {str(e)}\n¿Aseguraste que Ollama está corriendo y el nombre del modelo es correcto?"

def test_tools_model():
    print("="*50)
    print("🤖 PRUEBA 1: LAYA TOOL ROUTER")
    print("="*50)
    
    agent_query = "Read the contents of the local file config.yaml and then search the web for its author."
    available_tools = [
        "Read a file from the local filesystem",
        "Search the web for information",
        "Send an email",
        "Calculate a math expression"
    ]
    
    prompt = f"""You are Laya, an intelligent tool router.
Select the best tool for the following query.

Query: {agent_query}
Available Tools:
{json.dumps(available_tools, indent=2)}

Respond with a JSON object containing 'selected_tool' and 'reasoning'."""

    print(f"Pregunta del Agente: {agent_query}")
    print("\nLlamando al modelo...")
    response = query_ollama(LAYA_TOOLS_MODEL, prompt)
    print("\nRespuesta del Modelo:")
    print(response)


def test_context_model():
    print("\n" + "="*50)
    print("🧠 PRUEBA 2: LAYA CONTEXT ROUTER")
    print("="*50)
    
    current_task = "I need to fix the database connection timeout error that happens when user logs in."
    available_contexts = [
        "Memory: User previously mentioned they use PostgreSQL on port 5432.",
        "File snippet: db_config.py containing the connection string.",
        "Memory: The user likes the color blue for the UI.",
        "File snippet: login_handler.py where the timeout exception is caught.",
        "Document: Instructions on how to deploy to AWS."
    ]
    
    prompt = f"""You are Laya, an intelligent context selector.
Given the current task, select the strictly necessary context pieces from the available contexts.

Task: {current_task}
Available Contexts:
{json.dumps(available_contexts, indent=2)}

Respond with a JSON object containing 'selected_contexts' (a list) and 'reasoning'."""

    print(f"Tarea Actual: {current_task}")
    print("\nLlamando al modelo...")
    response = query_ollama(LAYA_CONTEXT_MODEL, prompt)
    print("\nRespuesta del Modelo:")
    print(response)

if __name__ == "__main__":
    print("Iniciando pruebas de los modelos Laya fine-tuned...\n")
    print("⚠️ IMPORTANTE: Asegúrate de haber importado los modelos a Ollama")
    print(f"con los nombres: '{LAYA_TOOLS_MODEL}' y '{LAYA_CONTEXT_MODEL}'")
    print("o cambia las variables en este script.\n")
    
    test_tools_model()
    test_context_model()
