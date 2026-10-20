import socket
from urllib.parse import urlparse
from openai import OpenAI, APIConnectionError, APITimeoutError
from dotenv import load_dotenv
from agent import Agent

load_dotenv()

def is_server_reachable(url="http://localhost:3000", timeout=2.0):
    """Verifica rápidamente si el socket del servidor está abierto y escuchando."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 3000
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False

print("Mi primer agente de IA (Local)")

SERVER_URL = "http://localhost:3000"
if not is_server_reachable(SERVER_URL):
    print("⚠️ [AVISO] No se detecta conexión activa con http://localhost:3000.")
    print("   Asegúrate de que tu túnel SSH o servidor remoto estén activos.\n")

# 1. Apuntamos al router de tu contenedor Docker / servidor remoto con timeout
client = OpenAI(
    base_url=f"{SERVER_URL}/v1",
    api_key="no-key-needed",
    timeout=90.0  # Evita que se quede colgado indefinidamente
)

agent = Agent(client)

# --- BUCLE 1: Interacción con el usuario ---
while True:
    user_input = input("\nTú: ").strip()
    
    if not user_input:
        continue
    
    if user_input.lower() in ("salir", "exit", "bye", "sayonara"):
        print("Hasta luego!")
        break

    # Verificar si el servidor está alcanzable antes de procesar
    if not is_server_reachable(SERVER_URL):
        print("❌ Jarvis: No puedo responderte porque no hay conexión con el servidor (http://localhost:3000).")
        print("   Verifica que el túnel SSH siga conectado y vuelve a intentarlo.")
        continue
    
    # 1. Agregamos el mensaje del usuario al historial
    agent.messages.append({"role": "user", "content": user_input})

    # 2. Limpiar tools descubiertas del turno anterior
    agent.discovered_tools = []

    # --- BUCLE 2: Interacción Modelo <-> Herramientas ---
    while True:
        # 3. Obtener tools activas (core + descubiertas por search_tools)
        active_tools = agent.get_active_tools()

        nombres_tools = [t['function']['name'] for t in active_tools]
        print(f"  🛠️ [DEBUG] Tools disponibles: {nombres_tools}")

        # Filtrar mensajes con Laya Context Router
        filtered_messages = agent.get_filtered_messages(user_input)

        # 4. Preparamos los argumentos para la API
        kwargs = {
            "model": "Bonsai-27b-Q1_0",
            "messages": filtered_messages,
            "tools": active_tools
        }

        # Llamamos al modelo con captura de errores de conexión y timeout
        try:
            response = client.chat.completions.create(**kwargs)
        except (APIConnectionError, ConnectionRefusedError):
            print("\n❌ Jarvis: Se perdió la conexión con el servidor LLM (http://localhost:3000).")
            print("   Revisa el túnel SSH y vuelve a escribir tu mensaje.")
            break
        except APITimeoutError:
            print("\n❌ Jarvis: El servidor tardó demasiado en responder (tiempo de espera agotado).")
            print("   El servidor puede estar ocupado o saturado. Intenta de nuevo en unos segundos.")
            break
        except Exception as e:
            print(f"\n❌ Jarvis: Ocurrió un error inesperado al consultar el modelo: {e}")
            break
        
        # El procesamiento de la respuesta (ver si llamó tool o dio texto final)
        called_tool = agent.process_response(response)
        
        # Si NO se llamó ninguna herramienta, tenemos la respuesta final
        if not called_tool:
            print("Jarvis:", response.choices[0].message.content)
            break  # Vuelve a pedirte input en el bucle 1