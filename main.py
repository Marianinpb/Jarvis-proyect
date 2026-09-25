from openai import OpenAI
from dotenv import load_dotenv
from agent import Agent

load_dotenv()

print("Mi primer agente de IA (Local)")

# 1. Apuntamos al router de tu contenedor Docker
client = OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="no-key-needed"
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
    
    # 1. Agregamos el mensaje del usuario al historial
    agent.messages.append({"role": "user", "content": user_input})
    
    # 2. RAG de Herramientas: Laya decide la lista de herramientas permitidas
    # Aquí se imprimen los veredictos individuales gracias al cambio en agent.py
    active_tools = agent.get_relevant_tools(user_input)
    
    # --- NUEVO PRINT DE DEBUG ---
    nombres_tools = [t['function']['name'] for t in active_tools]
    print(f"  🛠️ [DEBUG] Enviando al modelo las herramientas: {nombres_tools if nombres_tools else 'Ninguna'}")
    
    # --- BUCLE 2: Interacción Modelo <-> Herramientas ---
    while True:
        # 3. Preparamos los argumentos para la API
        kwargs = {
            "model": "Bonsai-27b-Q1_0",
            "messages": agent.messages
        }
        
        # Solo inyectamos "tools" si Laya aprobó al menos una
        if active_tools:
            kwargs["tools"] = active_tools

        # Llamamos al modelo desempaquetando los argumentos dinámicos
        response = client.chat.completions.create(**kwargs)
        
        # El procesamiento de la respuesta (ver si llamó tool o dio texto final)
        called_tool = agent.process_response(response)
        
        # Si NO se llamó ninguna herramienta, tenemos la respuesta final
        if not called_tool:
            print("Jarvis:", response.choices[0].message.content)
            break # Rompe este bucle 2 y vuelve a pedirte input en el bucle 1