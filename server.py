# ==============================================================================
# server.py — Agent
# ==============================================================================

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import json
import glob
import requests
import re
import time
import asyncio
import random
import pandas as pd
import pypdf
from docx import Document
from bs4 import BeautifulSoup
import chromadb
from chromadb.utils import embedding_functions
from mlx_lm import load, stream_generate, generate
from mlx_lm.sample_utils import make_sampler

# ==========================================
# 1. CONFIGURATION
# ==========================================
MODEL_PATH = "./base_model"
ADAPTER_PATH = "./adapters_final"
CHAT_DIR = "./chats"
HOST = "127.0.0.1"
PORT = 8000
TAVILY_API_KEY = "tvly-your-key-here"

MAX_RAG_RESULTS = 5
CHUNK_SIZE = 1000
MAX_TOKENS = 4096
ALLOWED_EXTENSIONS = {'.py', '.js', '.html', '.md', '.txt', '.json', '.sql', '.csv', '.pdf', '.docx', '.go', '.rs', '.cpp'}

os.makedirs(CHAT_DIR, exist_ok=True)

app = FastAPI(title="Self-Improving Expert Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 2. LOAD AI
# ==========================================
print("⏳ Loading LLM...", flush=True)
model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
print("✅ LLM Loaded.", flush=True)

# ==========================================
# 3. UNIVERSAL LOADER
# ==========================================
class UniversalLoader:
    @staticmethod
    def read_file(filepath: str) -> str:
        ext = os.path.splitext(filepath)[1].lower()
        try:
            if ext == '.pdf':
                reader = pypdf.PdfReader(filepath)
                text = ""
                for page in reader.pages: text += page.extract_text() + "\n"
                return text
            elif ext == '.docx':
                doc = Document(filepath)
                return "\n".join([para.text for para in doc.paragraphs])
            elif ext == '.csv':
                df = pd.read_csv(filepath)
                return df.head(50).to_markdown(index=False)
            else:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except Exception as e: return f"[Error: {e}]"

loader = UniversalLoader()

# ==========================================
# 4. RAG ENGINE (With Long-Term Memory)
# ==========================================
class RAGEngine:
    def __init__(self):
        # Persistent storage: This saves data to ./rag_storage folder
        self.client = chromadb.PersistentClient(path="./rag_storage")
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        # Collection 1: Project Code (Read-Only context)
        self.project = self.client.get_or_create_collection(name="project_knowledge", embedding_function=self.embed_fn)
        
        # Collection 2: Learned Memory (Read-Write Preferences)
        self.memory = self.client.get_or_create_collection(name="learned_memory", embedding_function=self.embed_fn)

    def ingest_project(self, project_path: str):
        if not os.path.exists(project_path): return
        # Note: We wipe project knowledge on reload, but KEEP learned memory
        try: self.client.delete_collection("project_knowledge")
        except: pass
        self.project = self.client.get_or_create_collection("project_knowledge", embedding_function=self.embed_fn)

        ids, docs, metas = [], [], []
        for root, _, files in os.walk(project_path):
            if any(x in root for x in ['.git', 'node_modules', 'venv', '__pycache__']): continue
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    fp = os.path.join(root, f)
                    content = loader.read_file(fp)
                    if not content.strip(): continue
                    chunks = [content[i:i+CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
                    for i, chunk in enumerate(chunks):
                        ids.append(f"{f}_{i}")
                        docs.append(chunk)
                        metas.append({"source": f, "path": fp, "type": ext})
        if docs:
            for i in range(0, len(docs), 2000):
                self.project.add(ids=ids[i:i+2000], documents=docs[i:i+2000], metadatas=metas[i:i+2000])

    def learn(self, text: str, category: str):
        """Saves a rule/preference forever."""
        print(f"💾 Learning: {text[:40]}...", flush=True)
        self.memory.add(
            ids=[f"mem_{int(time.time())}_{random.randint(0,999)}"],
            documents=[text],
            metadatas=[{"category": category, "timestamp": time.time()}]
        )

    def recall(self, query: str) -> str:
        """Retrieves both Code Context AND Learned Rules."""
        context = ""
        
        # 1. Search Codebase
        if self.project.count() > 0:
            results = self.project.query(query_texts=[query], n_results=MAX_RAG_RESULTS)
            if results['documents']:
                context += "--- RELEVANT PROJECT FILES ---\n"
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i]
                    context += f"File: {meta['source']}\nSnippet:\n{doc}\n...\n\n"

        # 2. Search Learned Memory (Preferences)
        if self.memory.count() > 0:
            mem_results = self.memory.query(query_texts=[query], n_results=3)
            if mem_results['documents'] and mem_results['documents'][0]:
                context += "--- YOUR LEARNED PREFERENCES ---\n"
                for doc in mem_results['documents'][0]:
                    context += f"- {doc}\n"
                context += "\n"
        
        return context

rag = RAGEngine()

# ==========================================
# 5. TOOLS
# ==========================================
def tool_web_search(query: str) -> List[Dict]:
    print(f"🔎 Searching: {query}", flush=True)
    try:
        r = requests.post("https://api.tavily.com/search", json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 5}, timeout=10)
        return r.json().get("results", [])
    except: return []

def tool_visit_page(url: str) -> str:
    print(f"🌐 Scraper visiting: {url}", flush=True)
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.content, 'html.parser')
        for script in soup(["script", "style", "nav", "footer"]): script.extract()
        text = soup.get_text(separator=' ')
        return text[:5000]
    except: return "Error reading page."

# ==========================================
# 6. AGENT BRAIN (With Reflection)
# ==========================================
class AgentBrain:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate_json(self, prompt: str) -> dict:
        full_prompt = f"<|im_start|>system\nOutput ONLY valid JSON.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        response = generate(self.model, self.tokenizer, prompt=full_prompt, max_tokens=400, verbose=False)
        try:
            text = response.replace("```json", "").replace("```", "").strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            return json.loads(match.group(0)) if match else json.loads(text)
        except: return {}

    def _generate_text(self, prompt: str) -> str:
        full_prompt = f"<|im_start|>system\nYou are an Expert Principal Engineer.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        response = generate(self.model, self.tokenizer, prompt=full_prompt, max_tokens=300, verbose=False)
        return response.strip()

    def detect_intent(self, query: str, file_context: str) -> str:
        # Check for explicit memory save commands
        if any(x in query.lower() for x in ["remember this", "save this rule", "save preference", "learn this"]):
            return "MEMORY_SAVE"

        prompt = f"""
        Request: "{query}"
        Has Files: {"Yes" if len(file_context) > 50 else "No"}

        Classify intent:
        1. "CODING": Wants code written, fixed, or modified.
        2. "GENERAL": General questions (weather, stocks, history).
        
        Return JSON: {{ "category": "CODING" }} or {{ "category": "GENERAL" }}
        """
        result = self._generate_json(prompt)
        return result.get("category", "GENERAL")

    def reflect_and_learn(self, last_user_msg: str, last_ai_msg: str):
        """Analyzes interaction to extract rules."""
        prompt = f"""
        TASK: Extract a RULE or PREFERENCE from this interaction.
        USER: "{last_user_msg}"
        AI: "{last_ai_msg[:500]}..."
        
        Did the user define a constraint? (e.g. "Don't use print", "Always use PyTorch")
        If YES, output the rule.
        If NO, output "NO_LEARNING".
        """
        insight = self._generate_text(prompt)
        if "NO_LEARNING" not in insight and len(insight) > 5:
            rag.learn(insight, category="preference")
            return insight
        return None

    def analyze_codebase(self, files_context: str, history_str: str) -> str:
        return self._generate_text(f"Analyze context. Files: {files_context[:2000]} History: {history_str[-2000:]}")

    def refine_query(self, user_query: str, analysis: str, history_str: str) -> str:
        return self._generate_text(f"Reformulate '{user_query}' into search query based on {analysis}. If 'add another', find NEW source.")

    def plan(self, refined_query: str, analysis: str) -> dict:
        if "NO_SEARCH_NEEDED" in refined_query: return {"needs_research": False, "steps": []}
        return self._generate_json(f"Goal: {refined_query}. Context: {analysis}. JSON Plan {{ needs_research: bool, steps: [{{tool, query}}] }}")

brain = AgentBrain(model, tokenizer)

# ==========================================
# 7. WORKFLOW
# ==========================================
class FileAttachment(BaseModel):
    name: str
    content: str

class ChatRequest(BaseModel):
    chat_id: Optional[str] = None
    message: str
    history: list = []
    project_path: str = ""
    attached_files: List[FileAttachment] = []
    use_search: bool = False

async def analytic_agent_stream(req: ChatRequest):
    chat_id = req.chat_id or str(int(time.time()))
    if req.project_path: rag.ingest_project(req.project_path)
    
    history_str = "\n".join([f"User: {h[0]}\nAI: {h[1]}" for h in req.history[-6:]])
    
    # --- CONTEXT LOADING (Includes Learned Memory!) ---
    files_context = ""
    if req.attached_files:
        files_context += "\n--- ATTACHED FILES ---\n"
        for f in req.attached_files:
            files_context += f"Filename: {f.name}\nContent:\n{f.content[:4000]}\n---\n"
    elif req.project_path:
        # Search BOTH Code and Memory for relevant info
        files_context = rag.recall(req.message + " " + "config setup")

    # --- PHASE 0: DETECT INTENT ---
    intent = brain.detect_intent(req.message, files_context)
    yield f'<div class="thinking-badge">🧭 Intent: {intent}</div>\n'
    await asyncio.sleep(0.02)

    # --- SPECIAL: MEMORY SAVE INTENT ---
    if intent == "MEMORY_SAVE":
        yield '<div class="thinking-badge">💾 Saving to Long-Term Memory...</div>\n'
        # Save the specific user instruction as a hard rule
        rag.learn(req.message, category="user_rule")
        yield "🧠 **Memory Updated.** I have saved this preference/rule to my permanent knowledge base."
        return # Stop processing

    # --- PHASE 1: ANALYZE ---
    yield '<div class="thinking-badge">🧐 Analyzing Context & Memory...</div>\n'
    await asyncio.sleep(0.02)
    project_analysis = brain.analyze_codebase(files_context, history_str)

    # --- PHASE 2: REFINE ---
    yield '<div class="thinking-badge">🤔 Synthesizing...</div>\n'
    await asyncio.sleep(0.02)
    refined_query = brain.refine_query(req.message, project_analysis, history_str)

    # --- PHASE 3: PLAN ---
    yield '<div class="thinking-badge">🧠 Planning...</div>\n'
    await asyncio.sleep(0.02)
    plan = brain.plan(refined_query, project_analysis)
    
    # --- PHASE 4: EXECUTE ---
    context_accumulated = f"ANALYSIS:\n{project_analysis}\n\n"
    
    if plan.get("needs_research", False) or req.use_search:
        steps = plan.get("steps", [])
        for i, step in enumerate(steps):
            tool = step.get("tool")
            query = step.get("query")
            
            yield f'<div class="thinking-badge">⚙️ Step {i+1}: {tool}("{query}")</div>\n'
            await asyncio.sleep(0.02)
            
            step_result = ""
            if tool == "web_search":
                results = await asyncio.to_thread(tool_web_search, query)
                snippet_summary = "\n".join([f"- {r['content']}" for r in results])
                step_result += f"Search Results:\n{snippet_summary}\n"
                if results:
                     best_url = results[0].get("url")
                     yield f'<div class="thinking-badge">🌐 Reading: {best_url}</div>\n'
                     await asyncio.sleep(0.02)
                     page_content = await asyncio.to_thread(tool_visit_page, best_url)
                     step_result += f"Page Content:\n{page_content}\n"
            elif tool == "read_project_files":
                step_result = rag.recall(query) # Use Recall to get memory + files
            
            context_accumulated += f"\n--- STEP {i+1} DATA ---\n{step_result}\n"

    # --- PHASE 5: ANSWER ---
    if intent == "CODING":
        system_prompt = """You are an Expert Senior Software Engineer.
Your task is to write or modify code based on the User Request.

STRICT RESPONSE STRUCTURE:
1. **Intro**: Start with a single normal text sentence (e.g., "Here is the updated script...").
2. **The Code**: Output the FULL code in a single Markdown block (e.g., ```python ... ```).
3. **The Summary**: AFTER closing the code block, add a section starting with "## 📝 Summary" and explain in 2-3 sentences what you changed and why.

RULES:
1. **CHECK MEMORY**: Use the "LEARNED PREFERENCES" in the context if available.
2. **NO LAZY DUPLICATION**: Do not simply duplicate existing variables. 
3. **COMPLETE CODE**: Output the FULL, runnable script. 
4. **MARKDOWN**: Ensure you CLOSE the code block (```) before writing the Summary.
"""
    else:
        system_prompt = """You are an Expert Researcher.
Your task is to answer the user's question comprehensively using the SEARCH DATA provided.

RULES:
1. **BE ACCURATE**: Base your answer on the provided search context.
2. **NO CODE BLOCKS FOR TEXT**: Do NOT use code blocks (```) for regular text.
3. **FORMATTING**: Use bolding, lists, and tables.
4. **NO HALLUCINATIONS**: If the search results don't contain the answer, state that clearly.
"""

    final_prompt = f"""<|im_start|>system
{system_prompt}
CONTEXT (Files & Learned Memory):
{files_context}
{project_analysis}
SEARCH DATA:
{context_accumulated}
CHAT HISTORY:
{history_str}
<|im_end|>
<|im_start|>user
{req.message}
<|im_end|>
<|im_start|>assistant
"""
    
    full_response_text = ""
    sampler = make_sampler(temp=0.7, top_p=0.95)
    
    for chunk in stream_generate(model, tokenizer, prompt=final_prompt, max_tokens=MAX_TOKENS, sampler=sampler):
        text = chunk.text
        full_response_text += text
        if "<|im_end|>" in text: break
        yield text

    # --- BACKGROUND LEARNING ---
    # If the user seems happy, we might want to auto-learn the last solution
    # Note: We do this implicitly. For explicit learning, user uses "Save this" intent.
    if any(x in req.message.lower() for x in ["great", "perfect", "good job", "thanks"]):
        if len(req.history) > 0:
            last_ai_msg = req.history[-1][1]
            asyncio.create_task(asyncio.to_thread(brain.reflect_and_learn, req.message, last_ai_msg))

    # SAVE HISTORY
    try:
        new_history = req.history + [[req.message, full_response_text]]
        chat_data = {
            "id": chat_id,
            "title": req.message[:30],
            "timestamp": time.time(),
            "history": new_history
        }
        with open(os.path.join(CHAT_DIR, f"{chat_id}.json"), "w", encoding="utf-8") as f:
            json.dump(chat_data, f)
    except: pass

@app.post("/api/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(analytic_agent_stream(req), media_type="text/plain")

@app.get("/api/history")
def get_history():
    chats = []
    for fp in glob.glob(f"{CHAT_DIR}/*.json"):
        try:
            with open(fp) as f: chats.append(json.load(f))
        except: pass
    return sorted(chats, key=lambda x: x.get("timestamp", 0), reverse=True)

@app.get("/api/history/{chat_id}")
def get_chat_detail(chat_id: str):
    path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    raise HTTPException(status_code=404, detail="Chat not found")

@app.delete("/api/history/{chat_id}")
def delete_chat(chat_id: str):
    path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    if os.path.exists(path): os.remove(path)
    return {"status": "deleted"}

if __name__ == "__main__":
    print(f"AGENT running on http://{HOST}:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)
