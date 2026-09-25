from laya import Router
router = Router(preload=True)
state = {"question": "puedes decirme que hay en main.py?", "passage": "Reads and returns the content of a specific file."}
questions = {
    "relevance": {"type": "score", "instructions": "How relevant is this passage?", "criteria": ["irrelevant", "weak", "useful", "direct answer"]},
    "keep": {"type": "noul", "instructions": "Should this passage be kept?"}
}
print(router.predict(state, questions))