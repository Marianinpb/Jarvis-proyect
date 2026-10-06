from laya import Router

# Rutas a tus repositorios en HuggingFace
TOOLS_MODEL_PATH = "Marianinpb/laya-toolrag-router"
CONTEXT_MODEL_PATH = "Marianinpb/laya-context-router"

def test_tools_model():
    print("="*50)
    print(" PRUEBA 1: LAYA TOOL ROUTER")
    print("="*50)
    
    try:
        router = Router(models={"english": TOOLS_MODEL_PATH}, default="english", preload=True)
    except Exception as e:
        print(f"Error cargando el modelo de Tools: {e}")
        return

    state = {
        "agent_query": "Read the contents of the local file config.yaml and then search the web for its author.",
        "available_tools": [
            "Read a file from the local filesystem",
            "Search the web for information",
            "Send an email",
            "Calculate a math expression"
        ]
    }
    
    questions = {
        "selected_tool": {
            "type": "choice", 
            "instructions": "Which tool should be selected for this query?", 
            "criteria": state["available_tools"]
        }
    }
    
    print(f"Consulta: {state['agent_query']}")
    print("Evaluando herramientas...")
    try:
        result = router.predict(state, questions)
        print(f"\nRespuesta del Modelo:\n{result}")
    except Exception as e:
        print(f"Error evaluando: {e}")


def test_context_model():
    print("\n" + "="*50)
    print(" PRUEBA 2: LAYA CONTEXT ROUTER")
    print("="*50)
    
    try:
        router = Router(models={"english": CONTEXT_MODEL_PATH}, default="english", preload=True)
    except Exception as e:
        print(f"Error cargando el modelo de Context: {e}")
        return

    state = {
        "current_task": "I need to fix the database connection timeout error that happens when user logs in.",
        "available_contexts": [
            "Memory: User previously mentioned they use PostgreSQL on port 5432.",
            "File snippet: db_config.py containing the connection string.",
            "Memory: The user likes the color blue for the UI.",
            "File snippet: login_handler.py where the timeout exception is caught.",
            "Document: Instructions on how to deploy to AWS."
        ]
    }
    
    print(f"Tarea: {state['current_task']}")
    print("Evaluando contextos (relevancia)...")
    
    for i, context in enumerate(state["available_contexts"]):
        eval_state = {
            "current_task": state["current_task"],
            "context_piece": context
        }
        questions = {
            "is_relevant": {
                "type": "noul", 
                "instructions": "Is this context piece strictly necessary to solve the current task?"
            }
        }
        res = router.predict(eval_state, questions)
        print(f"Contexto {i+1}: {context[:30]}... -> {res}")


if __name__ == "__main__":
    test_tools_model()
    test_context_model()
