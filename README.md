# cyber-expert-llm

**Build datasets, fine-tune, and evaluate AI models that think like a CISO, a cybersecurity expert, and a cybersecurity engineer.**

Works with any open model on Hugging Face (Llama, Qwen, Mistral, Gemma, Phi, and others) and exports chat JSONL that most fine-tuning tools and hosted services accept. It also ships three ready-to-use **Claude Skills**, so you can get the personas in Claude without training anything.

```
 taxonomy ──► generate ──► clean ──► human review ──► build ──► train (LoRA/QLoRA) ──► merge/serve ──► eval
 (topics)    (teacher)   (gates)    (flagged.jsonl)  (splits,   (TRL + PEFT)          (vLLM/Ollama)   (MCQ + judged
                                                      formats)                                          scenarios)
```

## The three personas

| Persona | Focus | Example asks |
|---|---|---|
| **CISO** (`ciso`) | Risk, governance, board reporting, regulation, budget, programme | "Restructure my board report", "100-day plan", "Do we file an 8-K?" |
| **Cybersecurity expert** (`expert`) | Incident response, threat intel, ATT&CK, architecture, threat modelling | "Ransomware: first 24 hours", "BEC investigation", "STRIDE this feature" |
| **Cybersecurity engineer** (`engineer`) | Detections, cloud/K8s hardening, IAM, CI/CD, vuln management | "Sigma + KQL for encoded PowerShell", "Block public S3 org-wide", "Review this workflow" |

Persona system prompts live in [`configs/personas.yaml`](configs/personas.yaml) and are attached at export time, so you can change a persona without regenerating data.

## What's in the box

```
configs/
  personas.yaml        system prompts for the three personas
  taxonomy.yaml        persona → domain → subtopics + frameworks (NIST CSF 2.0, ISO 27001, ATT&CK, ...)
  train_lora.yaml      LoRA/QLoRA training config
data/seed/             hand-written, reviewed example conversations (format reference + seed data)
evals/
  mcq.jsonl            multiple-choice knowledge questions with answers
  scenarios.jsonl      open-ended scenarios with grading rubrics, including safety checks
skills/                Claude Skills: ciso-advisor, security-expert, security-engineer
src/cyberllm/          the toolkit (CLI: `cyberllm`)
tests/                 pytest suite (no GPU or API key needed)
```

## Quickstart

```bash
git clone https://github.com/<you>/cyber-expert-llm && cd cyber-expert-llm
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
cyberllm validate 'data/seed/*.jsonl'
```

### 1. Generate synthetic data with a teacher model

Serve a teacher model behind any OpenAI-compatible endpoint (Ollama, vLLM, LM Studio, llama.cpp server, TGI, LiteLLM):

```bash
ollama serve &                    # or: vllm serve <model> --port 8000
ollama pull <teacher-model>
```

```bash
cyberllm generate --teacher openai:<teacher-model> --base-url http://localhost:11434/v1 -n 1500 --seed 1
```

The generator samples the taxonomy evenly (persona × domain × subtopic × task type × difficulty), and about 8% of the samples are **boundary** examples, where the model learns to decline harmful requests and still give useful defensive help. Runs can be resumed: re-running skips IDs already written. Use `--dry-run` to write the prompts without calling a model.

#### Choosing a teacher model

Only use a teacher whose **licence and terms allow its outputs to be used to train other models.** Many commercial APIs, including Anthropic's and OpenAI's, restrict this. For that reason `cyberllm generate` refuses to use Claude as a teacher; Claude is supported as an evaluation judge and as a baseline instead. Permissively licensed open-weight models (many are Apache-2.0) are the usual choice. Check the model card, and record which teacher you used in your dataset card.

### 2. Clean, review, and build

```bash
cyberllm clean data/seed/*.jsonl data/generated/synthetic.jsonl
#   kept      -> data/processed/clean.jsonl
#   flagged   -> data/review/flagged.jsonl   (safety lint hits: a human decides)
#   dropped   -> data/review/dropped.tsv     (schema, length, boilerplate, near-duplicates)

cyberllm build data/processed/clean.jsonl --format messages --format sharegpt
#   data/processed/{train,val,test}.{messages,sharegpt}.jsonl
```

- Splits are deterministic (hash of the record ID), so adding data never moves old records between train and test.
- `--persona ciso` builds a single-persona model; the default trains one model that handles all three, selected by system prompt.
- `--reviewed-only` keeps only seed, human, and synthetic records marked `"reviewed": true`. Use this for anything you publish.

### 3. Fine-tune (LoRA / QLoRA)

```bash
pip install -e '.[train]'
cyberllm train --config configs/train_lora.yaml
cyberllm train --set base_model=meta-llama/Llama-3.1-8B-Instruct --set num_train_epochs=2
```

With 4-bit QLoRA, a 7–8B model trains on a single 24 GB GPU, and roughly 16 GB can work with shorter `max_length`. The `messages` export also works directly with Axolotl, Unsloth, LLaMA-Factory, and hosted fine-tuning services that take chat JSONL. Check each provider's terms for your use case.

### 4. Serve your model

```bash
# Serve the LoRA adapter directly
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora --lora-modules cyber=outputs/cyber-expert-lora

# ...or merge the adapter into the base model first
cyberllm merge --adapter outputs/cyber-expert-lora --out outputs/cyber-expert-merged
vllm serve outputs/cyber-expert-merged --served-model-name cyber
```

### 5. Evaluate

```bash
pip install -e '.[claude]'          # Claude as the judge
export ANTHROPIC_API_KEY=...

cyberllm eval --target openai:cyber --base-url http://localhost:8000/v1 --name finetuned
cyberllm eval --target openai:Qwen/Qwen2.5-7B-Instruct --base-url http://localhost:8001/v1 --name base
cyberllm eval --target anthropic --name claude-baseline
```

Each run writes `reports/<name>.md` and `reports/<name>.json`:
- **MCQ accuracy**, overall and per persona.
- **Scenario scores** from 0 to 100%, graded by an LLM judge against per-item rubrics and split into **quality** and **safety**. A good model scores high on both: it's useful, and it declines operational attack help while offering the defensive alternative.

The judge can also be any OpenAI-compatible model (`--judge openai:<model> --judge-base-url ...`). Always compare **base vs fine-tuned** on the same items, and grow `evals/` with questions your own experts write. Never train on the eval files.

## Claude Skills (no training required)

`skills/` contains three Claude Skills that capture the same personas:

- **Claude Code**: copy a folder into `~/.claude/skills/` (personal) or `.claude/skills/` (project).
  ```bash
  cp -r skills/* ~/.claude/skills/
  ```
- **Claude apps**: zip a skill folder and upload it as a custom skill.

They're a fast way to get value today, and a useful baseline to beat with your fine-tuned model.

## Dataset record format

```json
{
  "id": "engineer-seed-0001",
  "persona": "ciso | expert | engineer",
  "domain": "detection-engineering",
  "subtopic": "writing a Sigma rule and converting it to KQL or SPL",
  "frameworks": ["Sigma", "MITRE ATT&CK"],
  "difficulty": "foundational | intermediate | advanced",
  "task_type": "advisory | explain | analyze | design | write | review | boundary",
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "source": "seed | synthetic | human",
  "reviewed": true
}
```

Multi-turn conversations are supported: alternate user and assistant turns, ending with an assistant turn. Don't put a system message in the record; it's added at export.

## Safety and responsible use

This project is for **defensive** security expertise.
- Training data teaches the model to explain attacks so defenders can detect and stop them, and to decline working malware, credential theft, phishing kits, and EDR evasion while still offering the legitimate alternative.
- `cyberllm clean` runs a heuristic safety lint (shellcode bytes, credential-dumping commands, AV/EDR tampering, ransom notes, and similar). Flagged records go to human review and are never silently kept. A lint isn't a guarantee: review samples before publishing.
- The eval suite includes safety scenarios, so regressions show up as numbers.
- Synthetic data can contain confident errors. Have practitioners review a sample of every batch, prefer `--reviewed-only` for releases, and treat model output as advice to verify, not as authority.

## Roadmap ideas

- [ ] Multi-turn generation (follow-up questions, pushback, clarifications)
- [ ] Retrieval-augmented variant grounded in framework texts with citations
- [ ] DPO/ORPO preference data from judge comparisons
- [ ] Larger expert-written eval sets per persona and per framework
- [ ] Dataset card and model card templates

Contributions welcome, especially **expert-reviewed** conversations and eval items. Run `pytest -q` and `cyberllm validate` before opening a PR.

## License

MIT. Datasets you generate are subject to your teacher model's licence; models you train are subject to your base model's licence.
