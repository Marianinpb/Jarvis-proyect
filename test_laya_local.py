import os
from laya import Router

# Rutas a tus repositorios en HuggingFace y local
TOOLS_MODEL_PATH = "Marianinpb/laya-toolrag-router"
CONTEXT_MODEL_PATH = "Marianinpb/laya-context-router"
LOCAL_MEMORY_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laya_memory_router")

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


# ═════════════════════════════════════════════════════════════════════
# PRUEBA 3: MEMORY ROUTER (relevante + guardar + categoría) sobre pares
# ═════════════════════════════════════════════════════════════════════
BASE_MODEL_PATH = None  # None => modelo base de Laya (convaiinnovations/laya)

STORE_THRESHOLD = 0.5
RELEVANT_THRESHOLD = 0.5

MEMORY_CATEGORIES = {
    "user_preferences": "Identity memory: immutable rules, personal user preferences, language, formatting or style instructions",
    "project_context": "Working memory: technical details of the user's current task or project (files, configs, errors, decisions)",
    "general_knowledge": "Semantic memory: concepts, definitions, general facts or reusable code snippets",
    "conversational_context": "Episodic memory: recent conversational thread, greetings, small talk, acknowledgements",
}

MEMORY_QUESTIONS = {
    "is_relevant": {
        "type": "noul",
        "instructions": "Is this exchange strictly necessary to solve the current task?",
    },
    "should_store": {
        "type": "noul",
        "instructions": "Does this exchange contain at least one piece of information worth saving in long-term memory for future conversations?",
    },
    "memory_type": {
        "type": "choice",
        "instructions": "Which type of memory does the most important information in this exchange belong to?",
        "criteria": MEMORY_CATEGORIES,
    },
}

# Cada caso: tarea actual + par (user, assistant) + etiquetas esperadas
MEMORY_CASES = [
    # ── user_preferences ──
    {"task": "Write a SQL query to count active users.",
     "user": "From now on, always answer me in Spanish and keep code comments in English.",
     "assistant": "Understood, I will answer in Spanish and keep comments in English.",
     "relevant": False, "store": True, "category": "user_preferences"},
    {"task": "Format the report summary.",
     "user": "I hate bullet points, please always use short paragraphs.",
     "assistant": "Got it, no bullet points from now on.",
     "relevant": True, "store": True, "category": "user_preferences"},
    {"task": "Refactor the login function.",
     "user": "Never use tabs, I only use 4 spaces for indentation.",
     "assistant": "Noted, 4 spaces it is.",
     "relevant": True, "store": True, "category": "user_preferences"},
    {"task": "Plan my week.",
     "user": "My name is Mariano and I work night shifts, so mornings are for sleeping.",
     "assistant": "Thanks Mariano, I'll keep your night schedule in mind.",
     "relevant": True, "store": True, "category": "user_preferences"},

    # ── project_context ──
    {"task": "Fix the database connection timeout on login.",
     "user": "The Postgres server runs on port 5433, not the default one.",
     "assistant": "Thanks, I will use port 5433 in the connection string.",
     "relevant": True, "store": True, "category": "project_context"},
    {"task": "Add a new endpoint to the Jarvis agent.",
     "user": "Jarvis runs Bonsai-27b through a local router on localhost:3000.",
     "assistant": "Ok, the agent client points to http://localhost:3000/v1.",
     "relevant": True, "store": True, "category": "project_context"},
    {"task": "Write a birthday message for my mom.",
     "user": "We decided to store tool schemas in tools_schema.json.",
     "assistant": "Right, dynamic tools are persisted in tools_schema.json.",
     "relevant": False, "store": True, "category": "project_context"},
    {"task": "Debug the failing unit test in agent.py.",
     "user": "Run the tests.",
     "assistant": "test_search_tools failed: KeyError 'probabilities' at line 331.",
     "relevant": True, "store": False, "category": "project_context"},

    # ── general_knowledge ──
    {"task": "Explain how my context filter works.",
     "user": "What is temperature scaling in classifiers?",
     "assistant": "It divides logits by a learned scalar T to calibrate probabilities without changing the argmax.",
     "relevant": False, "store": True, "category": "general_knowledge"},
    {"task": "Write a function to flatten nested lists in Python.",
     "user": "How do I flatten a list of lists?",
     "assistant": "Use: [x for sub in lst for x in sub]",
     "relevant": True, "store": True, "category": "general_knowledge"},
    {"task": "Book a flight to Madrid.",
     "user": "What does RAG stand for?",
     "assistant": "Retrieval-Augmented Generation: retrieving documents and feeding them to the LLM as context.",
     "relevant": False, "store": True, "category": "general_knowledge"},

    # ── conversational_context / ruido ──
    {"task": "Fix the database connection timeout on login.",
     "user": "Hi Jarvis, how are you?",
     "assistant": "Hello! I'm doing great, how can I help?",
     "relevant": False, "store": False, "category": "conversational_context"},
    {"task": "Refactor the login function.",
     "user": "ok thanks",
     "assistant": "You're welcome!",
     "relevant": False, "store": False, "category": "conversational_context"},
    {"task": "Refactor the login function.",
     "user": "Now continue with the next step we discussed.",
     "assistant": "Sure, next I'll extract the password validation into its own function.",
     "relevant": True, "store": False, "category": "conversational_context"},
    {"task": "Write a SQL query to count active users.",
     "user": "What's the weather in Madrid?",
     "assistant": "It is sunny and 25°C in Madrid.",
     "relevant": False, "store": False, "category": "conversational_context"},

    # ── "una sola línea valiosa" enterrada en un par largo ──
    {"task": "Summarize today's meeting.",
     "user": "lol yeah that movie was terrible, the ending made no sense, anyway btw my email is mariano@uni.mx for the reports, and did you see the game?",
     "assistant": "Haha, yes the ending was weird! I didn't watch the game. I'll send the reports to mariano@uni.mx.",
     "relevant": False, "store": True, "category": "user_preferences"},
    {"task": "Add logging to the agent.",
     "user": "ugh long day, coffee machine broke again... oh and remember the HF repo for the context model is Marianinpb/laya-context-router. anyway what's up",
     "assistant": "Sorry about the coffee! Noted the repo Marianinpb/laya-context-router. Not much, ready to help.",
     "relevant": False, "store": True, "category": "project_context"},
]


def _pair_state(case):
    return {
        "current_task": case["task"],
        "exchange": {"user": case["user"], "assistant": case["assistant"]},
    }


def test_memory_router(model_path, label):
    print("\n" + "=" * 70)
    print(f" PRUEBA 3: MEMORY ROUTER  [{label}]")
    print("=" * 70)

    try:
        if model_path:
            router = Router(models={"english": model_path}, default="english", preload=True)
        else:
            router = Router(default="english", preload=True)
    except Exception as e:
        print(f"Error cargando {label}: {e}")
        return None

    hits = {"is_relevant": 0, "should_store": 0, "memory_type": 0}
    confusion = {c: {c2: 0 for c2 in MEMORY_CATEGORIES} for c in MEMORY_CATEGORIES}

    for i, case in enumerate(MEMORY_CASES, 1):
        try:
            res = router.predict(_pair_state(case), MEMORY_QUESTIONS)
        except Exception as e:
            print(f"  ⚠️ Caso {i}: error {e!r}")
            continue
        ans = res.get("answers", {})
        p_rel = ans.get("is_relevant", {}).get("noul", 0.0)
        p_sto = ans.get("should_store", {}).get("noul", 0.0)
        cat = ans.get("memory_type", {}).get("choice")
        p_cat = ans.get("memory_type", {}).get("probabilities", {}).get(cat, 0.0)

        ok_rel = (p_rel > RELEVANT_THRESHOLD) == case["relevant"]
        ok_sto = (p_sto > STORE_THRESHOLD) == case["store"]
        ok_cat = cat == case["category"]
        hits["is_relevant"] += ok_rel
        hits["should_store"] += ok_sto
        hits["memory_type"] += ok_cat
        if cat in confusion:
            confusion[case["category"]][cat] += 1

        m = lambda ok: "[OK]" if ok else "[FAIL]"
        print(f"\n  [{i:02d}] task: {case['task'][:50]}")
        print(f"       user: {case['user'][:70]}")
        print(f"       {m(ok_rel)} relevante p={p_rel:.2f} (esperado {case['relevant']})"
              f" | {m(ok_sto)} guardar p={p_sto:.2f} (esperado {case['store']})"
              f" | {m(ok_cat)} {cat} p={p_cat:.2f} (esperado {case['category']})")

    n = len(MEMORY_CASES)
    print(f"\n  ── RESUMEN [{label}] ──")
    for q, h in hits.items():
        print(f"    {q:<14} {h}/{n} = {h / n:.0%}")
    print("\n    Matriz de confusión (fila=esperado, col=predicho):")
    short = {c: c[:6] for c in MEMORY_CATEGORIES}
    print("    " + " " * 24 + " ".join(f"{short[c]:>7}" for c in MEMORY_CATEGORIES))
    for c, row in confusion.items():
        print(f"    {c:<24}" + " ".join(f"{row[c2]:>7}" for c2 in MEMORY_CATEGORIES))
    return {q: h / n for q, h in hits.items()}


if __name__ == "__main__":
    import sys
    if "--old" in sys.argv:
        test_tools_model()
        test_context_model()
    else:
        results = {}
        if os.path.exists(LOCAL_MEMORY_MODEL_PATH):
            results["local-memory"] = test_memory_router(LOCAL_MEMORY_MODEL_PATH, "local memory router")
        results["fine-tuned"] = test_memory_router(CONTEXT_MODEL_PATH, "fine-tuned context")
        results["base"] = test_memory_router(BASE_MODEL_PATH, "Laya base")
        print("\n" + "=" * 70)
        print(" COMPARATIVA FINAL")
        print("=" * 70)
        for name, r in results.items():
            if r:
                print(f"  {name:<12} " + "  ".join(f"{q}={v:.0%}" for q, v in r.items()))

