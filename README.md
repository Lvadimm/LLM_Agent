Technical User Guide
Introduction
This project is a Local LLM (Large Language Model) Agent designed specifically for Apple Silicon hardware. Unlike standard chatbots, this system is "self-improving." It uses a feedback loop where your conversations become future training data, allowing the model to adapt to your specific coding style and preferences over time.

It is built on the MLX framework and uses a RAG architecture to combine its internal knowledge with external files and web search.

1. Prerequisites: The "Search" Capability
The agent uses a "Tool Use" pattern. When it doesn't know an answer, it can choose to browse the web. To enable this, it requires an API key for the Tavily search engine, which is optimized for LLMs.

Why Tavily? Unlike Google, Tavily returns clean text chunks that the AI can read easily, rather than full HTML pages full of ads.

Setup:

Get a key from tavily.com.

Open server.py.

Paste the key into the variable TAVILY_API_KEY. This allows the tool_web_search function inside the server to execute.

2. Phase 1: The Build Pipeline
Before you can chat, you must build the model. This pipeline moves from raw data to a specialized, fine-tuned model.

Step 1: Data Ingestion

Script: python 01_clean_download.py

What it does: This script utilizes the Hugging Face datasets library to pull two high-quality coding datasets: CodeAlpaca and Evol-Instruct.

Technical Context: It normalizes these distinct datasets into a single "ChatML" format ({"messages": [{"role": "user"}, {"role": "assistant"}]}). It also performs a randomized 90/10 split, saving a validation set so the training process can measure its accuracy as it learns.

Step 2: Model Acquisition

Script: python 02_prepare_dataset.py

What it does: Downloads the Qwen 2.5 Coder 7B model.

Technical Context: We fetch the 4-bit quantized version of the model. Quantization compresses the model weights so that a 7-billion parameter model can fit comfortably into the RAM of a MacBook (requiring only ~6-8GB of unified memory) without losing significant intelligence.

Step 3: LoRA Fine-Tuning (The "Learning" Step)

Script: ./03_train.sh

What it does: This initiates the training process using the MLX-LM library.

Technical Context: Instead of retraining the entire model (which is impossible on a laptop), this uses LoRA (Low-Rank Adaptation). LoRA freezes the main model and only trains a tiny set of "adapter layers" on top of it.

Optimization: The script is configured with batch-size 1 and grad-accumulation-steps 48. This simulates a much larger computer, allowing for stable training on consumer hardware.

Result: The output is not a huge model file, but small "adapter" files in adapters_final that represent the new knowledge.

Step 4: Integration Test

Script: python 04_merge_and_test.py

What it does: It loads the base model and dynamically "fuses" your new adapters at runtime to generate a sample response.

Technical Context: It uses mlx_lm.generate to run inference. If the output code (a FastAPI app example) looks correct, it confirms that your adapters have successfully modified the model's behavior.

3. Phase 2: Running the System
This system is composed of two parts: a FastAPI Backend (the brain) and a React Frontend (the interface).

The Launcher

Script: ./start_app.command

How it works: This is a Bash script that orchestrates the startup. It acts as a process manager:

Starts the Python server in the background.

Waits 10 seconds to ensure the Neural Networks are fully loaded into memory.

Navigates to the Q folder and starts the React frontend (npm run dev).

Safety: It includes a trap command, which ensures that if you close the terminal, it automatically kills both the Python and Node.js processes so they don't drain your battery.

The Backend "Brain"

File: server.py

Architecture: This is a FastAPI application that serves as the central nervous system.

Key Components:

RAG Engine (ChromaDB): It uses a vector database to index your files. This allows the AI to "read" your codebase by finding relevant snippets and injecting them into the chat context.

Long-Term Memory: It has a specific database collection (learned_memory) where it stores rules you teach it (e.g., "Always use TypeScript").

Intent Detection: Before answering, the AgentBrain analyzes your request to decide if it needs to write code, search the web, or just chat.

4. Phase 3: The Active Learning Loop
This is the feature that makes this agent unique. It does not just consume data; it creates it.

Script: python export_to_trainer.py

The Concept: As you use the tool, the server.py saves every conversation as a JSON file in the chats directory.

The Process: This script parses those logs. It filters out short or "broken" conversations and formats the high-quality interactions into the JSONL training format.

Closing the Loop: By pointing the training script (03_train.sh) to this new data file, you effectively "bake" your recent experiences into the model's permanent weights. This is how the agent aligns with your coding style over weeks of usage.

Quick File Reference
File	Type	Role & Technical Function
01_clean_download.py	Python	ETL Pipeline: Extract, Transform, Load script for dataset normalization.
02_prepare_dataset.py	Python	Model Fetcher: Downloads 4-bit quantized weights from Hugging Face.
03_train.sh	Shell	Trainer: MLX LoRA training script with memory optimization flags.
04_merge_and_test.py	Python	Unit Test: Verifies inference and adapter fusion.
server.py	Python	API Server: FastAPI backend handling RAG, search tools, and context management.
export_to_trainer.py	Python	Data Converter: Transforms chat logs into training datasets.
start_app.command	Shell	Orchestrator: Startup script for managing Backend (Python) and Frontend (React) processes.
