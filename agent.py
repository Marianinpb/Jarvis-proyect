import os
import json
import warnings
import importlib

# Silenciar barras de progreso y advertencias internas de Laya
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")

# Si el archivo tools.py no existe, lo creamos vacío
if not os.path.exists("tools.py"):
    with open("tools.py", "w", encoding="utf-8") as f:
        f.write("# Archivo autogenerado para el cerebro extendido de Jarvis\n")

import tools  # Importamos el archivo externo de herramientas dinámicas
from laya import Router

class Agent:
    def __init__(self, client):
        self.client = client
        self.setup_tools()
        self.messages = [
            {"role": "system", "content": (
                "Eres Jarvis, un asistente de IA avanzado. "
                "Tu ÚNICA herramienta inicial es search_tools. "
                "SIEMPRE que el usuario te pida algo, PRIMERO llama search_tools con una descripción "
                "EN INGLÉS de lo que necesitas hacer. Ejemplo: si te piden listar archivos, llama "
                "search_tools(query='list files in directory'). Si te piden la hora, llama "
                "search_tools(query='get current system time'). "
                "search_tools te devolverá las herramientas disponibles que podrás usar en tu siguiente turno. "
                "NUNCA digas que no puedes hacer algo sin antes buscar con search_tools."
            )}
        ]
        
        print("Cargando motor lógico de Jarvis...")
        self.laya_router = Router(models={"english": "Marianinpb/laya-toolrag-router"}, default="english", preload=True)

        # Tools descubiertas por search_tools en el turno actual
        self.discovered_tools = []
    
    def setup_tools(self):
        # ═══════════════════════════════════════════════════════════════
        # CORE TOOLS: SOLO search_tools, siempre en contexto del LLM
        # ═══════════════════════════════════════════════════════════════
        self.core_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_tools",
                    "description": "Searches the agent's memory for tools and recipes. Write the query IN ENGLISH. Example: 'list files in directory'. Returns matching tools you can use in your next response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "English description of the capability you need. Be specific."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # ═══════════════════════════════════════════════════════════════
        # BUILT-IN TOOLS: siempre disponibles pero solo vía search_tools
        # ═══════════════════════════════════════════════════════════════
        self.built_in_tools = [
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
                    "description": "Edits an existing text file by replacing old text with new text, or creates a simple text file. Do NOT use this to create agent tools or recipes.",
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
            },
            {
                "type": "function",
                "function": {
                    "name": "create_tool",
                    "description": "Teaches the agent a new permanent Python skill by writing executable code. Use ONLY when the user wants the agent to learn a new capability via code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Name of the new function."},
                            "description": {"type": "string", "description": "Clear description in English of what the tool does."},
                            "parameters": {"type": "object", "description": "JSON schema of the function parameters."},
                            "code": {"type": "string", "description": "Python code defining the function. Must start with 'def function_name(...):'"}
                        },
                        "required": ["name", "description", "parameters", "code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_recipe",
                    "description": "Saves a reusable multi-step workflow (recipe) that chains existing tools together. Use when the user wants to save a routine so the agent can repeat it without re-reasoning.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Unique name for the recipe (snake_case)."},
                            "description": {"type": "string", "description": "Clear English description of what the recipe does."},
                            "parameters": {"type": "object", "description": "JSON schema of input parameters the recipe needs from the user."},
                            "steps": {
                                "type": "array",
                                "description": "Ordered list of steps. Each step has: tool (tool name), args (dict with values or {param_name} placeholders or {step_N} to reference output of step N).",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "tool": {"type": "string"},
                                        "args": {"type": "object"}
                                    },
                                    "required": ["tool", "args"]
                                }
                            }
                        },
                        "required": ["name", "description", "steps"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_recipe",
                    "description": "Executes a previously saved recipe by name, filling in the provided parameters. Use when the user wants to repeat a saved workflow.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "recipe_name": {"type": "string", "description": "Name of the saved recipe to run."},
                            "params": {"type": "object", "description": "Input values to fill into the recipe's parameter placeholders."}
                        },
                        "required": ["recipe_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_terminal_command",
                    "description": "Executes a command in the system terminal (command prompt/shell) and returns the output. Use this to run scripts, install packages, or perform system operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to execute in the terminal."
                            }
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Performs a web search to find information on the internet. Returns a list of relevant URLs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        # ═══════════════════════════════════════════════════════════════
        # DYNAMIC REGISTRY: herramientas creadas con create_tool
        # (solo accesibles vía search_tools, no enviadas al LLM directamente)
        # ═══════════════════════════════════════════════════════════════
        self.dynamic_tools = []
        if os.path.exists("tools_schema.json"):
            with open("tools_schema.json", "r", encoding="utf-8") as f:
                try:
                    self.dynamic_tools = json.load(f)
                except json.JSONDecodeError:
                    pass

        # Cargar recetas previas
        self.recipes = {}
        if os.path.exists("recipes.json"):
            with open("recipes.json", "r", encoding="utf-8") as f:
                try:
                    recipe_list = json.load(f)
                    for r in recipe_list:
                        self.recipes[r["name"]] = r
                except json.JSONDecodeError:
                    pass

    # ─────────────────────────────────────────────────────────────────────
    # get_active_tools: lo que el LLM puede usar AHORA
    # ─────────────────────────────────────────────────────────────────────
    def get_active_tools(self):
        """Core tools + cualquier tool descubierta por search_tools en este turno."""
        return self.core_tools + self.discovered_tools

    # ─────────────────────────────────────────────────────────────────────
    # search_tools: el LLM busca capacidades extra (LAYA recibe inglés)
    # ─────────────────────────────────────────────────────────────────────
    def search_tools(self, query: str):
        """
        El LLM llama esto con query EN INGLÉS.
        LAYA choice clasifica contra dynamic_tools + recetas.
        Las herramientas encontradas se inyectan para el siguiente turno del LLM.
        """
        print(f"  🔍 Ejecutando: search_tools('{query}')")

        # Construir catálogo: built-in + dynamic tools + recetas
        searchable = []
        desc_to_item = {}

        for tool in self.built_in_tools + self.dynamic_tools:
            desc = tool["function"]["description"]
            desc_to_item[desc] = {"type": "tool", "schema": tool}
            searchable.append(desc)

        for name, recipe in self.recipes.items():
            desc = recipe.get("description", name)
            desc_to_item[desc] = {"type": "recipe", "name": name, "recipe": recipe}
            searchable.append(desc)

        if not searchable:
            return {"found": [], "message": "No hay herramientas ni recetas en la memoria extendida."}

        searchable.append("none of the above")

        state = {"question": query}
        questions = {
            "match": {
                "type": "choice",
                "instructions": "Which capability best matches what is needed?",
                "criteria": searchable
            }
        }

        try:
            result = self.laya_router.predict(state, questions)
            match_ans = result.get("answers", {}).get("match", {})
            probabilities = match_ans.get("probabilities", {})
        except Exception as e:
            print(f"  ⚠️ [LAYA] Error en search_tools: {repr(e)}")
            return {"found": [], "error": str(e)}

        THRESHOLD = 0.05
        found = []
        
        # Restablecemos la lista en cada iteración de búsqueda para no acumular
        self.discovered_tools = []

        for desc, prob in probabilities.items():
            if desc == "none of the above":
                continue
            if prob > THRESHOLD:
                item = desc_to_item.get(desc)
                if item and item["type"] == "tool":
                    tool_schema = item["schema"]
                    tool_name = tool_schema["function"]["name"]
                    # Inyectar para que el LLM pueda usarla en el siguiente turno
                    if tool_schema not in self.discovered_tools:
                        self.discovered_tools.append(tool_schema)
                    print(f"   ✅ Tool encontrada: {tool_name} (prob={prob:.4f})")
                    found.append({"type": "tool", "name": tool_name, "description": desc})
                elif item and item["type"] == "recipe":
                    r_name = item["name"]
                    print(f"   ✅ Receta encontrada: {r_name} (prob={prob:.4f})")
                    found.append({"type": "recipe", "name": r_name, "description": desc})
            else:
                print(f"   ❌ {desc[:50]}... (prob={prob:.4f})")

        if found:
            return {"found": found, "message": f"Se encontraron {len(found)} capacidades. Ya puedes usarlas."}
        else:
            return {"found": [], "message": "No se encontró ninguna herramienta o receta que coincida."}

    # ─────────────────────────────────────────────────────────────────────
    # Implementaciones de herramientas core
    # ─────────────────────────────────────────────────────────────────────
    def get_weather(self, location):
        print(f"  ⚙️ Ejecutando: get_weather({location})")
        return {"clima": "Soleado", "temperatura": "25°C"}
    
    def list_files_in_dir(self, directory="."):
        print(f"  ⚙️ Ejecutando: list_files_in_dir({directory})")
        try:
            files = os.listdir(directory)
            return {"files": files}
        except Exception as e:
            return {"error": str(e)}
        
    def read_file(self, path):
        print(f"  ⚙️ Ejecutando: read_file({path})")
        try:
            with open(path, encoding="utf-8") as f:
                return {"content": f.read()}
        except Exception as e:
            err = f"Error al leer el archivo {path}: {str(e)}"
            return {"error": err}
        
    def edit_file(self, path, prev_text="", new_text=""):
        print(f"  ⚙️ Ejecutando: edit_file({path})")
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

    def create_tool(self, name, description, parameters, code):
        print(f"  ⚙️ Ejecutando: create_tool({name})")
        try:
            # 1. Escribir el código físicamente en tools.py
            with open("tools.py", "a", encoding="utf-8") as f:
                f.write(f"\n\n{code}\n")
                
            # 2. Recompilar el archivo en tiempo real
            importlib.reload(tools)
            
            # Comprobar que realmente se compiló bien
            if not hasattr(tools, name):
                return {"error": f"El código se guardó pero no se detectó la función '{name}'."}

            # 3. Guardar el esquema para que sobreviva a los reinicios
            new_schema = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters if isinstance(parameters, dict) else json.loads(parameters)
                }
            }
            
            # Deduplicar por nombre
            self.dynamic_tools = [s for s in self.dynamic_tools if s.get("function", {}).get("name") != name]
            self.dynamic_tools.append(new_schema)
            with open("tools_schema.json", "w", encoding="utf-8") as f:
                json.dump(self.dynamic_tools, f, indent=4)
            
            return {"status": f"Herramienta '{name}' creada y registrada. Usa search_tools('{description}') para encontrarla."}
        except Exception as e:
            return {"error": f"Error creando la herramienta permanente: {str(e)}"}

    def create_recipe(self, name, description, steps, parameters=None):
        """Guarda una receta: workflow multi-paso reutilizable."""
        print(f"  ⚙️ Ejecutando: create_recipe({name})")
        try:
            recipe = {
                "name": name,
                "description": description,
                "parameters": parameters or {},
                "steps": steps if isinstance(steps, list) else json.loads(steps)
            }

            # Guardar en memoria
            self.recipes[name] = recipe

            # Persistir a disco
            all_recipes = list(self.recipes.values())
            with open("recipes.json", "w", encoding="utf-8") as f:
                json.dump(all_recipes, f, indent=4, ensure_ascii=False)

            return {"status": f"Receta '{name}' guardada con {len(recipe['steps'])} pasos."}
        except Exception as e:
            return {"error": f"Error creando receta: {str(e)}"}

    def run_recipe(self, recipe_name, params=None):
        """
        Ejecuta una receta guardada paso a paso.
        Soporta placeholders:
          - {nombre_param}: se reemplaza con params["nombre_param"]
          - {step_N}: se reemplaza con el resultado del paso N (0-indexed)
        """
        print(f"  ⚙️ Ejecutando: run_recipe({recipe_name})")
        params = params or {}

        if recipe_name not in self.recipes:
            return {"error": f"Receta '{recipe_name}' no encontrada. Recetas disponibles: {list(self.recipes.keys())}"}

        recipe = self.recipes[recipe_name]
        step_results = []

        for i, step in enumerate(recipe["steps"]):
            tool_name = step["tool"]
            raw_args = step.get("args", {})

            # Resolver placeholders en los argumentos
            resolved_args = {}
            for key, val in raw_args.items():
                if isinstance(val, str):
                    for pname, pval in params.items():
                        val = val.replace(f"{{{pname}}}", str(pval))
                    for j, prev_result in enumerate(step_results):
                        result_str = prev_result if isinstance(prev_result, str) else json.dumps(prev_result)
                        val = val.replace(f"{{step_{j}}}", result_str)
                resolved_args[key] = val

            print(f"    📋 Paso {i+1}/{len(recipe['steps'])}: {tool_name}({resolved_args})")

            if hasattr(self, tool_name):
                result = getattr(self, tool_name)(**resolved_args)
            elif hasattr(tools, tool_name):
                result = getattr(tools, tool_name)(**resolved_args)
            else:
                result = {"error": f"Tool '{tool_name}' no encontrada en paso {i+1}"}
                step_results.append(result)
                break

            step_results.append(result)
            print(f"    ✅ Paso {i+1} completado.")

        return {"status": f"Receta '{recipe_name}' ejecutada ({len(step_results)} pasos).", "results": step_results}

    def run_terminal_command(self, command):
        print(f"  ⚙️ Ejecutando: run_terminal_command({command})")
        try:
            import subprocess
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except Exception as e:
            return {"error": str(e)}

    def web_search(self, query):
        print(f"  ⚙️ Ejecutando: web_search({query})")
        try:
            try:
                from googlesearch import search
            except ImportError:
                return {"error": "El paquete 'googlesearch-python' no está instalado. Ejecuta 'pip install googlesearch-python' en la terminal (usando run_terminal_command o manualmente) para usar esta herramienta."}
            
            results = list(search(query, num_results=5, lang="es"))
            return {"results": results}
        except Exception as e:
            return {"error": str(e)}
        
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
                    elif fn_name == "create_tool":
                        result = self.create_tool(**args)
                    elif fn_name == "create_recipe":
                        result = self.create_recipe(**args)
                    elif fn_name == "run_recipe":
                        result = self.run_recipe(**args)
                    elif fn_name == "search_tools":
                        result = self.search_tools(**args)
                    elif fn_name == "run_terminal_command":
                        result = self.run_terminal_command(**args)
                    elif fn_name == "web_search":
                        result = self.web_search(**args)
                    # RUTEADOR DINÁMICO: tools descubiertas o de tools.py
                    elif hasattr(tools, fn_name):
                        try:
                            func = getattr(tools, fn_name) 
                            result = func(**args)
                        except Exception as e:
                            result = {"error": f"Error ejecutando {fn_name} desde tools.py: {str(e)}"}
                    else:
                        result = {"error": f"Función {fn_name} desconocida"}
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            
            return True
            
        return False

