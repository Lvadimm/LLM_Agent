import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkGfm from 'remark-gfm';
import { motion, AnimatePresence } from 'framer-motion';
import { Copy, Check, Bot, User, ChevronDown, ChevronRight, BrainCircuit, Terminal } from 'lucide-react';
import 'highlight.js/styles/atom-one-dark.css';

// --- SUB-COMPONENT: CODE BLOCK (Fixes Copy & State Issues) ---
const CodeBlock = ({ inline, className, children, ...props }) => {
  const [copied, setCopied] = useState(false);
  const codeRef = useRef(null); // Ref to access the actual DOM node

  if (inline) {
    return (
      <code {...props} className="bg-white/10 text-orange-200 px-1.5 py-0.5 rounded text-[12px] font-mono border border-white/5 break-words">
        {children}
      </code>
    );
  }

  const match = /language-(\w+)/.exec(className || '');
  const lang = match ? match[1].toUpperCase() : 'TEXT';

  // Render text blocks as simple boxes (no terminal styling)
  if (['TEXT', 'PLAINTEXT', 'TXT', 'MARKDOWN', 'MD'].includes(lang)) {
    return (
      <div className="bg-[#252526] p-4 rounded-lg border border-white/5 text-gray-300 font-mono text-[13px] my-4 whitespace-pre-wrap leading-relaxed shadow-sm">
        {children}
      </div>
    );
  }

  const handleCopy = () => {
    if (codeRef.current) {
      // Get the raw text from the DOM element, ignoring HTML tags/React objects
      const text = codeRef.current.innerText || codeRef.current.textContent;
      navigator.clipboard.writeText(text.replace(/\n$/, ''));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="relative group my-6 rounded-xl overflow-hidden border border-white/10 shadow-2xl bg-[#0d0d0d]">
      {/* Header */}
      <div className="flex justify-between items-center bg-[#1a1a1a] px-4 py-2 border-b border-white/5 select-none">
        <div className="flex items-center gap-2">
          <Terminal size={12} className="text-gray-500" />
          <span className="text-[10px] font-bold text-gray-500 uppercase tracking-wider">{lang}</span>
        </div>
        <button 
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-[10px] text-gray-400 hover:text-white transition-colors uppercase font-medium bg-white/5 hover:bg-white/10 px-2 py-1 rounded"
        >
          {copied ? <Check size={11} className="text-emerald-400"/> : <Copy size={11} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      
      {/* Code Area - Note the 'ref={codeRef}' */}
      <div className="overflow-x-auto custom-scrollbar">
        <code 
          ref={codeRef} 
          {...props} 
          className={`${className} !bg-transparent !p-4 block text-[13px] leading-loose font-mono tab-4`}
        >
          {children}
        </code>
      </div>
    </div>
  );
};

// --- MAIN COMPONENT: MESSAGE ---
export const Message = ({ role, content }) => {
  const [isExpanded, setIsExpanded] = useState(true); 
  
  // 1. Parsing Thinking Blocks
  const thoughtRegex = /<div class="thinking-badge">(.*?)<\/div>/g;
  const thoughts = [];
  let match;
  while ((match = thoughtRegex.exec(content)) !== null) {
    thoughts.push(match[1]);
  }
  
  const cleanContent = content.replace(thoughtRegex, '').trim();

  useEffect(() => {
    if (cleanContent.length > 20) setIsExpanded(false);
  }, [cleanContent.length]);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex w-full mb-8 ${role === 'user' ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`flex max-w-[95%] md:max-w-[85%] ${role === 'user' ? 'flex-row-reverse' : 'flex-row'} gap-4`}>
        
        {/* AVATAR */}
        <div className={`flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center shadow-lg border border-white/5 ${
          role === 'user' 
            ? 'bg-gradient-to-br from-blue-600 to-blue-700' 
            : 'bg-[#252526]'
        }`}>
          {role === 'user' ? <User size={16} className="text-white" /> : <Bot size={18} className="text-emerald-500" />}
        </div>

        {/* CONTENT */}
        <div className={`flex flex-col min-w-0 w-full ${role === 'user' ? 'items-end' : 'items-start'}`}>
          
          {/* Thinking Process */}
          {role === 'assistant' && thoughts.length > 0 && (
            <div className="mb-4 w-full max-w-xl">
              <button 
                onClick={() => setIsExpanded(!isExpanded)}
                className="group flex items-center gap-2 text-[11px] font-medium text-gray-500 hover:text-gray-300 transition-colors bg-[#1e1e1e] border border-white/5 px-3 py-2 rounded-lg w-full hover:bg-[#252526]"
              >
                <div className="p-1 rounded bg-blue-500/10 text-blue-400 group-hover:text-blue-300">
                    <BrainCircuit size={13} />
                </div>
                <span>Reasoning Process</span>
                <span className="text-gray-700">•</span>
                <span className="text-gray-500">{thoughts.length} steps</span>
                <div className="ml-auto text-gray-600 group-hover:text-gray-400">
                    {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </div>
              </button>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div 
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="pt-3 pl-2 flex flex-col gap-2">
                      {thoughts.map((thought, i) => (
                        <div key={i} className="flex gap-3 items-start text-[11px] text-gray-400 font-mono pl-3 border-l border-white/10 relative pb-1">
                          <div className={`absolute -left-[3px] top-[6px] w-1.5 h-1.5 rounded-full ${i === thoughts.length - 1 && !cleanContent ? 'bg-emerald-500 animate-pulse' : 'bg-[#333]'}`}></div>
                          <div className="leading-relaxed break-words [&>a]:text-blue-400 [&>a]:underline hover:[&>a]:text-blue-300" dangerouslySetInnerHTML={{ __html: thought }} />
                        </div>
                      ))}
                      {!cleanContent && (
                        <div className="flex gap-2 items-center pl-5 mt-1 text-[11px] text-emerald-500/80 animate-pulse font-mono uppercase tracking-wider">
                           <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span>
                           Processing...
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          {/* Response Text/Code */}
          {cleanContent && (
             <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className={`text-[14px] leading-7 w-full overflow-hidden ${
                  role === 'user' 
                    ? 'bg-blue-600 text-white rounded-2xl rounded-tr-sm px-5 py-3 shadow-lg' 
                    : 'text-gray-300'
                }`}
            >
                <ReactMarkdown 
                  remarkPlugins={[remarkGfm]} 
                  rehypePlugins={[rehypeHighlight]}
                  components={{
                    p: ({node, ...props}) => <p {...props} className={`mb-4 last:mb-0 leading-relaxed ${role === 'user' ? 'text-blue-50' : 'text-gray-300'}`} />,
                    a: ({...props}) => <a {...props} className="text-blue-400 hover:text-blue-300 hover:underline font-medium" target="_blank" rel="noopener noreferrer" />,
                    h1: ({...props}) => <h1 {...props} className="text-xl font-bold text-white mb-4 mt-6 pb-2 border-b border-white/10" />,
                    h2: ({...props}) => <div className="mt-8 pt-4 border-t border-white/10"><h2 {...props} className="text-lg font-bold text-white mb-3" /></div>,
                    h3: ({...props}) => <h3 {...props} className="text-base font-semibold text-gray-200 mb-2 mt-4" />,
                    ul: ({...props}) => <ul {...props} className="list-disc pl-6 mb-4 space-y-2 marker:text-gray-500" />,
                    ol: ({...props}) => <ol {...props} className="list-decimal pl-6 mb-4 space-y-2 marker:text-gray-500" />,
                    blockquote: ({...props}) => <blockquote {...props} className="border-l-4 border-blue-500/30 pl-4 py-1 my-4 text-gray-400 italic bg-white/5 rounded-r" />,
                    
                    table: ({node, ...props}) => (
                        <div className="overflow-x-auto my-6 border border-white/10 rounded-lg bg-[#1a1a1a] shadow-inner">
                            <table {...props} className="w-full text-left text-sm border-collapse" />
                        </div>
                    ),
                    thead: ({node, ...props}) => <thead {...props} className="bg-[#252526] text-gray-200 border-b border-white/10" />,
                    th: ({node, ...props}) => <th {...props} className="p-3 font-semibold whitespace-nowrap" />,
                    td: ({node, ...props}) => <td {...props} className="p-3 border-b border-white/5 text-gray-400 whitespace-nowrap" />,
                    
                    // Use the custom CodeBlock component here
                    code: CodeBlock 
                  }}
                >
                  {cleanContent}
                </ReactMarkdown>
            </motion.div>
          )}
        </div>
      </div>
    </motion.div>
  );
};