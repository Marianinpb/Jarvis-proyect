import json
import os

content_md_path = r'C:\Users\mtico\.gemini\antigravity-ide\brain\9ae38f14-be82-4c82-b6fb-dc5fdc050bda\.system_generated\steps\263\content.md'
with open(content_md_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

json_str = ''.join(lines[8:])
nb = json.loads(json_str)

# Limpiar las instrucciones de Kaggle y DDP en el primer markdown
nb['cells'][0]['source'] = [
    "# Fine-Tuning Laya on Custom Dataset (Colab 1x T4 GPU)\n",
    "\n",
    "Este cuaderno adapta el script oficial de Laya para entrenar ModernBERT (Laya) con 1 sola GPU T4 en Colab.\n"
]
# Quitar DDP check
nb['cells'][2]['source'] = [
    "!nvidia-smi\n",
    "import os, torch\n",
    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n"
]
# Quitar celdas DDP (DDP script block)
del nb['cells'][9]

# Reemplazar la celda de pre-procesamiento de datasets
dataset_cell = [
    "import os, json, torch\n",
    "from transformers import AutoTokenizer\n",
    "from huggingface_hub import snapshot_download\n",
    "from laya.agent import _fix_tokenizer_config\n",
    "from laya.common import build_sequence, render_options, QTYPES\n",
    "\n",
    "MODEL_ID = \"convaiinnovations/laya\"\n",
    "print(f\"Fetching tokenizer and config from {MODEL_ID}...\")\n",
    "model_dir = snapshot_download(MODEL_ID)\n",
    "_fix_tokenizer_config(model_dir)\n",
    "\n",
    "tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, \"tokenizer\"))\n",
    "with open(os.path.join(model_dir, \"rl_agent_config.json\")) as f:\n",
    "    cfg = json.load(f)\n",
    "\n",
    "print(\"Loading custom dataset from GDrive...\")\n",
    "\n",
    "def build_training_item(state, q, ans):\n",
    "    t = q[\"type\"]\n",
    "    crit = q.get(\"criteria\", {})\n",
    "    if t == \"choice\":\n",
    "        keys = list(crit.keys())\n",
    "        choice = ans.get(\"choice\")\n",
    "        target = [1.0 if k == choice else 0.0 for k in keys]\n",
    "    elif t == \"noul\":\n",
    "        val = ans.get(\"noul\")\n",
    "        target = [1.0 if not val else 0.0, 1.0 if val else 0.0]\n",
    "    elif t == \"score\":\n",
    "        n_levels = len(crit) if isinstance(crit, list) else 4\n",
    "        score_val = ans.get(\"score\")\n",
    "        target = [1.0 if str(i) == str(score_val) else 0.0 for i in range(n_levels)]\n",
    "    else:\n",
    "        return None\n",
    "    \n",
    "    s = sum(target)\n",
    "    target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)\n",
    "    label = target.index(max(target))\n",
    "    k = len(render_options({\"t\": t, \"crit\": crit}))\n",
    "    \n",
    "    seq, markers = build_sequence(tok, state, {\"t\": t, \"ins\": q[\"instructions\"], \"crit\": crit}, cfg[\"max_len\"], cfg[\"head_max_len\"])\n",
    "    if len(markers) != k:\n",
    "        return None\n",
    "    return {\n",
    "        \"ids\": seq,\n",
    "        \"markers\": markers,\n",
    "        \"qtype\": QTYPES[t],\n",
    "        \"target\": target,\n",
    "        \"label\": label\n",
    "    }\n",
    "\n",
    "items = []\n",
    "with open('/content/drive/MyDrive/laya_training_dataset.jsonl', 'r', encoding='utf-8') as f:\n",
    "    for row in f:\n",
    "        data = json.loads(row)\n",
    "        state = data[\"state\"]\n",
    "        questions = data[\"questions\"]\n",
    "        answers = data[\"answers\"]\n",
    "        for qid, q in questions.items():\n",
    "            if qid in answers:\n",
    "                it = build_training_item(state, q, answers[qid])\n",
    "                if it:\n",
    "                    items.append(it)\n",
    "\n",
    "print(f\"Preprocessed {len(items)} training sequences.\")\n",
    "torch.save(items, \"/content/train_items.pt\")\n",
    "print(\"Saved preprocessed items to /content/train_items.pt\")\n"
]
nb['cells'][6]['source'] = dataset_cell

# Escribir single GPU training cell replacing torchrun
training_cell = [
    "import os, sys, time, json, random, math\n",
    "import numpy as np\n",
    "import torch\n",
    "from safetensors.torch import load_file, save_file\n",
    "from transformers import AutoTokenizer\n",
    "from laya.common import build_model, proper_reward, QTYPES, TEMP_MIN, TEMP_MAX\n",
    "\n",
    "def collate_train_batch(items, pad_id):\n",
    "    n, L = len(items), max(len(it[\"ids\"]) for it in items)\n",
    "    kmax = max(len(it[\"markers\"]) for it in items)\n",
    "    ids = torch.full((n, L), pad_id, dtype=torch.long)\n",
    "    att = torch.zeros((n, L), dtype=torch.long)\n",
    "    mpos = torch.zeros((n, kmax), dtype=torch.long)\n",
    "    mmask = torch.zeros((n, kmax), dtype=torch.bool)\n",
    "    target = torch.zeros((n, kmax), dtype=torch.float32)\n",
    "    for i, it in enumerate(items):\n",
    "        ids[i, : len(it[\"ids\"])] = torch.tensor(it[\"ids\"])\n",
    "        att[i, : len(it[\"ids\"])] = 1\n",
    "        k = len(it[\"markers\"])\n",
    "        mpos[i, :k] = torch.tensor(it[\"markers\"])\n",
    "        mmask[i, :k] = True\n",
    "        target[i, : len(it[\"target\"])] = torch.tensor(it[\"target\"], dtype=torch.float32)\n",
    "    return {\n",
    "        \"input_ids\": ids,\n",
    "        \"attention_mask\": att,\n",
    "        \"marker_positions\": mpos,\n",
    "        \"marker_mask\": mmask,\n",
    "        \"targets\": target,\n",
    "    }\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "model_dir = snapshot_download(MODEL_ID)\n",
    "print(\"Building Laya ModernBERT model...\")\n",
    "model, cfg = build_model(model_dir)\n",
    "state_dict = load_file(os.path.join(model_dir, \"model.safetensors\"))\n",
    "model.load_state_dict(state_dict, strict=False)\n",
    "model.to(device)\n",
    "model.train()\n",
    "\n",
    "items = torch.load(\"/content/train_items.pt\")\n",
    "random.seed(42)\n",
    "random.shuffle(items)\n",
    "\n",
    "pad_id = AutoTokenizer.from_pretrained(os.path.join(model_dir, \"tokenizer\")).pad_token_id\n",
    "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)\n",
    "batch_size = 16\n",
    "\n",
    "for epoch in range(3):\n",
    "    total_loss = 0\n",
    "    random.shuffle(items)\n",
    "    for i in range(0, len(items), batch_size):\n",
    "        batch = collate_train_batch(items[i:i+batch_size], pad_id)\n",
    "        batch = {k: v.to(device) for k, v in batch.items()}\n",
    "        \n",
    "        logits, _ = model(\n",
    "            input_ids=batch[\"input_ids\"],\n",
    "            attention_mask=batch[\"attention_mask\"],\n",
    "            marker_positions=batch[\"marker_positions\"],\n",
    "            marker_mask=batch[\"marker_mask\"],\n",
    "        )\n",
    "        loss = torch.nn.functional.cross_entropy(logits[batch[\"marker_mask\"]], batch[\"targets\"][batch[\"marker_mask\"]])\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "        optimizer.zero_grad()\n",
    "        total_loss += loss.item()\n",
    "    print(f\"Epoch {epoch+1} | Loss: {total_loss / (len(items) / batch_size):.4f}\")\n",
    "\n",
    "print(\"Training Complete!\")\n",
    "os.makedirs(\"/content/laya_finetuned\", exist_ok=True)\n",
    "save_file(model.state_dict(), \"/content/laya_finetuned/model.safetensors\")\n",
    "print(\"Saved model to /content/laya_finetuned/model.safetensors\")\n"
]
# Replacing the bash execution for torchrun with pure Python
nb['cells'][8]['source'] = training_cell
nb['cells'][8]['cell_type'] = 'code'

# Replace any mention of kaggle paths with content
for cell in nb['cells']:
    if 'source' in cell:
        cell['source'] = [line.replace('/kaggle/working/', '/content/') for line in cell['source']]

with open('train_laya_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
print("Done parsing")
