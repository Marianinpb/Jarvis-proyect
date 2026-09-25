import os
import json
import warnings

# 1. Silenciar las barras de progreso y advertencias internas de Laya
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

from laya import Router

class Agent:
    def __init__(self, client):
        # client se guarda por si en el futuro se requiere para otra cosa
        self.client = client
        self.setup_tools()
        self.messages = [
            {"role": "system", "content": "Eres Jarvis, un asistente de IA avanzado. TIENES ACCESO a herramientas para listar, leer y editar archivos en la computadora del usuario. NUNCA digas que no puedes acceder al sistema. Si el usuario pide información sobre archivos, DEBES llamar a las herramientas proporcionadas."}
        ]
        
        print("Cargando motor lógico de Jarvis...")
        self.laya_router = Router(preload=True)
    
    def setup_tools(self):
        self.master_tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_files_in_dir",
                    "description": "Lists all files and folders in a given directory (defaults to current directory).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {
                                "type": "string",
                                "description": "Directory to list (optional). Defaults to current."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Reads and returns the content of a specific file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "The exact path of the file to read."
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Creates a new file or edits an existing one by replacing prev_text with new_text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path of the file to edit or create."
                            },
                            "prev_text": {
                                "type": "string",
                                "description": "Text to be replaced (can be empty for new files)."
                            },
                            "new_text": {
                                "type": "string",
                                "description": "New text to insert or replace."
                            }
                        },
                        "required": ["path", "new_text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Gets the current weather forecast for a city. Use ONLY if the user explicitly asks about the weather, temperature, or rain.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The name of the city."
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

    # ─────────────────────────────────────────────────────────────────────
    # ETAPA 1: LAYA choice — clasificador de intent multilingüe
    # ─────────────────────────────────────────────────────────────────────
    def classify_intent(self, user_query: str) -> str:
        """
        Usa LAYA `choice` (Router auto-detect -> multilingüe para español)
        para clasificar el intent del usuario y devolver el label en inglés
        correspondiente a la descripción de la tool más probable.

        Ventajas sobre el LLM:
        - Funciona en cualquier idioma sin traducción previa
        - Nunca devuelve vacío (choice siempre elige un label)
        - ~33 ms, determinista, sin costo adicional
        - El label devuelto ES la descripción en inglés de la tool -> listo para Etapa 2
        """
        # Criterios = descripciones de tools (inglés) + opción de "sin tool"
        criteria = [tool["function"]["description"] for tool in self.master_tools]
        criteria.append("general conversation, no tool needed")

        state = {"question": user_query}
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "Which capability does the user need?",
                "criteria": criteria
            }
        }

        try:
            # Sin model= -> Router detecta idioma -> usa multilingüe para español
            result = self.laya_router.predict(state, questions)


            intent_ans = result.get("answers", {}).get("intent", {})
            # LAYA choice devuelve {"choice": "...", "probabilities": {...}, ...}
            # (no "label" como en otros frameworks)
            if isinstance(intent_ans, dict):
                chosen = intent_ans.get("choice", "") or intent_ans.get("label", "")
            else:
                chosen = str(intent_ans)

            print(f"  [LAYA choice] Intent detectado: '{chosen}'")
            return chosen

        except Exception as e:
            print(f"  [LAYA choice] Error: {repr(e)}")
            return ""

    # ─────────────────────────────────────────────────────────────────────
    # ETAPA 2: LAYA score — ranker de relevancia en inglés
    # ─────────────────────────────────────────────────────────────────────
    def get_relevant_tools(self, user_query: str):
        """
        Pipeline de selección de herramientas con dos primitivas LAYA encadenadas:

        Etapa 1 — LAYA choice (multilingüe, auto-detect):
            Clasifica el intent en cualquier idioma.
            Devuelve el label en inglés (= descripción de la tool ganadora).

        Etapa 2 — LAYA score (inglés, model="laya"):
            Puntúa la relevancia de CADA tool contra ese intent en inglés.
            Scores reproducibles y consistentes con el playground en línea.
        """
        print("\n  [DEBUG] Etapa 1 — Clasificando intent con LAYA choice...")
        english_intent = self.classify_intent(user_query)

        if not english_intent or english_intent == "general conversation, no tool needed":
            print("  [LAYA] Conversación general — sin herramientas.")
            return []

        print(f"  [DEBUG] Etapa 2 — LAYA score contra: '{english_intent}'")
        relevant_tools = []

        # Rango [0.0, 3.0] con 4 criterios ordinales.
        # La Etapa 1 ya filtró el intent -> umbral conservador 1.2
        THRESHOLD = 1.2

        questions = {
            "relevance": {
                "type": "score",
                "instructions": "How relevant is this passage to answering the question?",
                "criteria": ["irrelevant", "weak", "useful", "direct answer"]
            }
        }

        for tool in self.master_tools:
            tool_name = tool["function"]["name"]
            tool_desc = tool["function"]["description"]

            passage = f"{tool_name}: {tool_desc}"
            # Etapa 2: question = intent en inglés -> model="laya" -> scores reproducibles
            state = {"question": english_intent, "passage": passage}
            prediction = self.laya_router.predict(state, questions, model="laya")

            respuestas      = prediction.get("answers", {})
            relevance_score = respuestas.get("relevance", {}).get("score", 0.0)

            if relevance_score > THRESHOLD:
                print(f"   APROBADA: {tool_name} (score={relevance_score:.4f})")
                relevant_tools.append(tool)
            else:
                print(f"   RECHAZADA: {tool_name} (score={relevance_score:.4f})")

        return relevant_tools

    # ─────────────────────────────────────────────────────────────────────
    # Implementaciones de herramientas
    # ─────────────────────────────────────────────────────────────────────
    def get_weather(self, location):
        print(f"  Ejecutando: get_weather({location})")
        return {"clima": "Soleado", "temperatura": "25°C"}
    
    def list_files_in_dir(self, directory="."):
        print(f"  Ejecutando: list_files_in_dir({directory})")
        try:
            files = os.listdir(directory)
            return {"files": files}
        except Exception as e:
            return {"error": str(e)}
        
    def read_file(self, path):
        print(f"  Ejecutando: read_file({path})")
        try:
            with open(path, encoding="utf-8") as f:
                return {"content": f.read()}
        except Exception as e:
            err = f"Error al leer el archivo {path}: {str(e)}"
            return {"error": err}
        
    def edit_file(self, path, prev_text="", new_text=""):
        print(f"  Ejecutando: edit_file({path})")
        try:
            existed = os.path.exists(path)
            if existed and prev_text:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                if prev_text not in content:
                    return {"error": f"Texto no encontrado: {prev_text}"}
                content = content.replace(prev_text, new_text)
            else:
                dir_name = os.path.dirname(path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                content = new_text
                
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
                
            action = "editado" if existed and prev_text else "creado"
            return {"status": f"Archivo {path} {action} exitosamente"}
        except Exception as e:
            err = f"Error al crear/editar {path}: {str(e)}"
            return {"error": err}
        
    def process_response(self, response):
        message = response.choices[0].message
        self.messages.append(message)
        
        if message.tool_calls:
            llamadas_procesadas = set()
            
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                raw_args = tool_call.function.arguments
                
                firma = f"{fn_name}_{raw_args}"
                
                if firma in llamadas_procesadas:
                    result = {"status": "Llamada ignorada por ser duplicado exacto"}
                else:
                    llamadas_procesadas.add(firma)
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    
                    if fn_name == "list_files_in_dir":
                        result = self.list_files_in_dir(**args)
                    elif fn_name == "read_file":
                        result = self.read_file(**args)
                    elif fn_name == "edit_file":
                        result = self.edit_file(**args)
                    elif fn_name == "get_weather":
                        result = self.get_weather(**args)
                    else:
                        result = {"error": f"Función {fn_name} desconocida"}
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            
            return True
            
        return False
