import React, { useState, useEffect, useRef } from 'react';
import { Send, Folder, File as FileIcon, Menu, Plus, StopCircle, Search, Bot, Trash2 } from 'lucide-react';
import { motion } from 'framer-motion';
import { Message } from './components/Message';

const API_URL = "http://127.0.0.1:8000/api";

function App() {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]); 
  const [chatId, setChatId] = useState(null); 
  const [isLoading, setIsLoading] = useState(false);
  const [useSearch, setUseSearch] = useState(false);
  const [files, setFiles] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const abortController = useRef(null);
  const messagesEndRef = useRef(null);

  // Initial History Load
  useEffect(() => { fetchHistory(); }, []);
  // Auto-scroll
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const fetchHistory = async () => {
    try {
      const res = await fetch(`${API_URL}/history`);
      const data = await res.json();
      setHistory(data);
    } catch (e) { console.error(e); }
  };

  const loadChat = async (id) => {
    if (chatId === id) return;

    try {
      const res = await fetch(`${API_URL}/history/${id}`);
      if (!res.ok) throw new Error("Chat not found");
      const data = await res.json();
      setChatId(data.id);
      
      const flatMsgs = [];
      data.history.forEach(([u, b]) => {
        flatMsgs.push({ role: 'user', content: u });
        flatMsgs.push({ role: 'assistant', content: b });
      });
      setMessages(flatMsgs);
      setFiles([]); 
    } catch (e) {
      console.error(e);
    }
  };

  const deleteChat = async (id, e) => {
    e.stopPropagation(); 
    if (!confirm("Are you sure you want to delete this chat?")) return;

    try {
      await fetch(`${API_URL}/history/${id}`, { method: "DELETE" });
      if (chatId === id) startNewChat();
      fetchHistory();
    } catch (e) {
      console.error("Failed to delete", e);
    }
  };

  const startNewChat = () => {
    setChatId(null);
    setFiles([]); 
    setMessages([{ role: 'assistant', content: "Hey! I'm your Agentic Assistant.\nI can research complex topics, verify libraries, and write code." }]);
    setInput("");
  };

  const handleSend = async () => {
    if (!input.trim() && files.length === 0) return;
    const userMsg = input;
    setInput("");
    
    // Add user message to UI immediately
    const newMessages = [...messages, { role: 'user', content: userMsg }];
    setMessages(newMessages);
    setIsLoading(true);

    // --- CRITICAL FIX START: ROBUST HISTORY GENERATION ---
    // The previous logic failed because the chat starts with an Assistant message (index 0).
    // This new logic explicitly finds "User" -> "Assistant" pairs to build history.
    const historyPayload = [];
    for (let i = 0; i < newMessages.length - 1; i++) {
        const currentMsg = newMessages[i];
        const nextMsg = newMessages[i+1];

        // We only care about completed turns: User said X, Assistant said Y.
        if (currentMsg.role === 'user' && nextMsg?.role === 'assistant') {
            historyPayload.push([currentMsg.content, nextMsg.content]);
        }
    }
    // --- CRITICAL FIX END ---

    // Add placeholder for AI response
    setMessages(prev => [...prev, { role: 'assistant', content: "" }]);
    abortController.current = new AbortController();

    try {
      let activeChatId = chatId;
      if (!activeChatId) {
        activeChatId = Date.now().toString(); 
        setChatId(activeChatId);
      }

      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
            chat_id: activeChatId,
            message: userMsg, 
            history: historyPayload, // Now correctly populated
            attached_files: files, 
            use_search: useSearch 
        }),
        signal: abortController.current.signal
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let botText = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        botText += chunk;
        
        setMessages(prev => {
          const updated = [...prev];
          updated[updated.length - 1].content = botText;
          return updated;
        });
      }
      
      fetchHistory(); 
      setFiles([]);

    } catch (error) { 
        if (error.name !== 'AbortError') console.error(error); 
    } finally { 
        setIsLoading(false); 
        abortController.current = null; 
    }
  };

  const handleFileUpload = (e) => {
    Array.from(e.target.files).forEach(file => {
      const reader = new FileReader();
      reader.onload = (ev) => setFiles(prev => [...prev, { name: file.name, content: ev.target.result }]);
      reader.readAsText(file);
    });
  };

  return (
    <div className="flex h-screen bg-[#1e1e1e] text-gray-300 font-sans overflow-hidden text-[13px]">
      
      {/* SIDEBAR */}
      <motion.div 
        initial={{ width: 260 }}
        animate={{ width: sidebarOpen ? 260 : 0 }}
        className="bg-[#181818] border-r border-white/5 flex flex-col flex-shrink-0"
      >
        <div className="p-3 border-b border-white/5">
          <button onClick={startNewChat} className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-blue-700 to-blue-600 hover:from-blue-600 hover:to-blue-500 text-white py-2.5 rounded-lg font-medium shadow-lg shadow-blue-900/20 transition-all active:scale-[0.98] text-xs">
            <Plus size={14} /> New Chat
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
          {history.map(chat => (
            <div 
              key={chat.id} 
              onClick={() => loadChat(chat.id)}
              className={`p-2.5 mb-1 rounded-md cursor-pointer flex justify-between items-center group transition-all text-xs ${chatId === chat.id ? 'bg-white/10 text-white font-medium' : 'hover:bg-white/5 text-gray-400'}`}
            >
              <span className="truncate pr-2">{chat.title}</span>
              <button 
                onClick={(e) => deleteChat(chat.id, e)}
                className="opacity-0 group-hover:opacity-100 text-gray-500 hover:text-red-400 p-1 rounded transition-all"
                title="Delete Chat"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      </motion.div>

      {/* MAIN AREA */}
      <div className="flex-1 flex flex-col relative min-w-0 bg-[#1e1e1e]">
        <div className="h-12 border-b border-white/5 flex items-center px-4 justify-between bg-[#1e1e1e]/80 backdrop-blur-md z-10">
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-1.5 hover:bg-white/5 rounded text-gray-400 hover:text-white transition-colors">
            <Menu size={16} />
          </button>
          <div className="flex items-center gap-2 text-xs font-medium opacity-40 uppercase tracking-widest">
            <Bot size={14} /> Agentic RAG
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 md:px-12 md:py-6 custom-scrollbar">
          {messages.map((msg, idx) => (
            <Message key={idx} role={msg.role} content={msg.content} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* INPUT AREA */}
        <div className="p-4 pt-2 bg-[#1e1e1e]">
          <div className="max-w-3xl mx-auto flex flex-col gap-2">
            
            {/* FILE PREVIEW */}
            {files.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {files.map((f, i) => (
                  <div key={i} className="flex items-center gap-1.5 text-[10px] bg-white/5 px-2 py-1 rounded border border-white/5 text-gray-300">
                    <FileIcon size={10} /> {f.name} 
                    <button onClick={() => setFiles(files.filter((_, fi) => fi !== i))} className="hover:text-red-400 ml-1">×</button>
                  </div>
                ))}
              </div>
            )}

            <div className="relative bg-[#252526] rounded-xl border border-white/10 shadow-2xl focus-within:border-blue-500/50 focus-within:ring-1 focus-within:ring-blue-500/20 transition-all">
              <textarea 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSend())}
                placeholder="Ask complex questions..."
                className="w-full bg-transparent text-gray-200 p-3 pr-28 outline-none resize-none h-[50px] max-h-[150px] text-[13px] placeholder:text-gray-600 custom-scrollbar"
              />
              
              <div className="absolute top-1.5 right-1.5 flex items-center gap-1">
                 <button 
                  onClick={() => setUseSearch(!useSearch)}
                  className={`p-1.5 rounded-lg transition-all ${useSearch ? 'bg-blue-500/20 text-blue-400' : 'text-gray-500 hover:bg-white/5 hover:text-gray-300'}`}
                  title="Force Deep Search"
                >
                  <Search size={14} />
                </button>
                
                <label className="p-1.5 text-gray-500 hover:bg-white/5 hover:text-gray-300 rounded-lg cursor-pointer transition-colors">
                  <input type="file" multiple className="hidden" onChange={handleFileUpload} />
                  <Folder size={14} />
                </label>

                {isLoading ? (
                  <button onClick={() => abortController.current?.abort()} className="p-1.5 bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 transition-colors">
                    <StopCircle size={14} />
                  </button>
                ) : (
                  <button 
                    onClick={handleSend} 
                    disabled={!input.trim() && files.length === 0} 
                    className="p-1.5 bg-gradient-to-br from-blue-600 to-blue-500 text-white rounded-lg hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-blue-500/20 transition-all active:scale-95"
                  >
                    <Send size={14} />
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;