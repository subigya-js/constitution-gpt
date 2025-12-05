'use client';

import { useEffect, useRef, useState } from 'react';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

const SUGGESTED_QUESTIONS = [
    "How is the Prime Minister elected in Nepal?",
    "What are the fundamental rights of citizens?",
    "What are the duties of citizens?",
    "How is the President elected?",
    "What is the structure of the Federal Parliament?",
    "What are the provisions for freedom of speech?"
];

export default function ChatInterface() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
            textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
        }
    }, [input]);

    const handleSend = async (question?: string) => {
        const messageContent = question || input.trim();
        if (!messageContent || isLoading) return;

        const userMessage: Message = {
            id: Date.now().toString(),
            role: 'user',
            content: messageContent,
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        // Simulate AI response (replace with actual API call)
        setTimeout(() => {
            const aiMessage: Message = {
                id: (Date.now() + 1).toString(),
                role: 'assistant',
                content: `This is a simulated response to: "${messageContent}"\n\nIn a production environment, this would connect to your RAG backend at the retrieval_pipeline.py endpoint to fetch constitutional information with proper citations and hierarchical structure.\n\n📘 Part 7 – Federal Executive\nArticle 76 – Constitution of Council of Ministers\n\n🔹 Sub-article (1)\nAs per Part 7, Article 76, Sub-article (1):\n• The President shall appoint the leader of a parliamentary party that commands a majority in the House of Representatives as the Prime Minister...`,
                timestamp: new Date(),
            };
            setMessages(prev => [...prev, aiMessage]);
            setIsLoading(false);
        }, 1500);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="flex flex-col h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-indigo-950 dark:to-purple-950">
            {/* Header */}
            <header className="glass border-b border-white/20 dark:border-white/10 px-6 py-4 backdrop-blur-xl">
                <div className="max-w-5xl mx-auto flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg">
                            <span className="text-white text-xl font-bold">🏛️</span>
                        </div>
                        <div>
                            <h1 className="text-xl font-bold gradient-text">Constitution GPT</h1>
                            <p className="text-xs text-gray-600 dark:text-gray-400">AI-Powered Constitutional Intelligence</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse-slow"></div>
                        <span className="text-xs text-gray-600 dark:text-gray-400">Online</span>
                    </div>
                </div>
            </header>

            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto px-4 py-6">
                <div className="max-w-4xl mx-auto space-y-6">
                    {messages.length === 0 ? (
                        <div className="flex flex-col items-center justify-center h-full space-y-8 animate-fade-in">
                            <div className="text-center space-y-4">
                                <div className="w-20 h-20 mx-auto rounded-2xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-2xl">
                                    <span className="text-4xl">🏛️</span>
                                </div>
                                <h2 className="text-3xl font-bold gradient-text">Welcome to Constitution GPT</h2>
                                <p className="text-gray-600 dark:text-gray-400 max-w-md">
                                    Ask any question about the Constitution of Nepal. Get accurate, citation-backed answers with proper hierarchical structure.
                                </p>
                            </div>

                            {/* Suggested Questions */}
                            <div className="w-full max-w-2xl space-y-3">
                                <p className="text-sm font-medium text-gray-700 dark:text-gray-300 text-center">Try asking:</p>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                    {SUGGESTED_QUESTIONS.map((question, index) => (
                                        <button
                                            key={index}
                                            onClick={() => handleSend(question)}
                                            className="group p-4 text-left rounded-xl bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm border border-gray-200/50 dark:border-gray-700/50 hover:border-indigo-500/50 dark:hover:border-indigo-400/50 transition-all duration-300 hover:shadow-lg hover:scale-[1.02] animate-slide-up"
                                            style={{ animationDelay: `${index * 0.1}s` }}
                                        >
                                            <div className="flex items-start gap-3">
                                                <span className="text-indigo-500 dark:text-indigo-400 mt-0.5 group-hover:scale-110 transition-transform">💡</span>
                                                <span className="text-sm text-gray-700 dark:text-gray-300 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                                                    {question}
                                                </span>
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    ) : (
                        <>
                            {messages.map((message, index) => (
                                <div
                                    key={message.id}
                                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in`}
                                >
                                    <div
                                        className={`max-w-[80%] rounded-2xl px-5 py-3 shadow-lg ${message.role === 'user'
                                            ? 'bg-gradient-to-br from-indigo-500 to-purple-600 text-white'
                                            : 'glass border border-gray-200/50 dark:border-gray-700/50'
                                            }`}
                                    >
                                        {message.role === 'assistant' && (
                                            <div className="flex items-center gap-2 mb-2 pb-2 border-b border-gray-200/50 dark:border-gray-700/50">
                                                <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
                                                    <span className="text-xs">🏛️</span>
                                                </div>
                                                <span className="text-xs font-semibold text-gray-700 dark:text-gray-300">Constitution GPT</span>
                                            </div>
                                        )}
                                        <div className={`text-sm leading-relaxed whitespace-pre-wrap ${message.role === 'user' ? 'text-white' : 'text-gray-800 dark:text-gray-200'
                                            }`}>
                                            {message.content}
                                        </div>
                                        <div className={`text-xs mt-2 ${message.role === 'user' ? 'text-indigo-100' : 'text-gray-500 dark:text-gray-400'
                                            }`}>
                                            {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                                        </div>
                                    </div>
                                </div>
                            ))}
                            {isLoading && (
                                <div className="flex justify-start animate-fade-in">
                                    <div className="glass border border-gray-200/50 dark:border-gray-700/50 rounded-2xl px-5 py-4 shadow-lg">
                                        <div className="flex items-center gap-2">
                                            <div className="flex gap-1">
                                                <div className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" style={{ animationDelay: '0s' }}></div>
                                                <div className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                                                <div className="w-2 h-2 rounded-full bg-pink-500 animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                                            </div>
                                            <span className="text-sm text-gray-600 dark:text-gray-400">Analyzing constitution...</span>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Input Area */}
            <div className="glass border-t border-white/20 dark:border-white/10 px-4 py-4 backdrop-blur-xl">
                <div className="max-w-4xl mx-auto">
                    <div className="relative flex items-end gap-3">
                        <div className="flex-1 relative">
                            <textarea
                                ref={textareaRef}
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Ask about the Constitution of Nepal..."
                                rows={1}
                                className="w-full px-5 py-4 pr-12 rounded-2xl bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 focus:border-indigo-500 dark:focus:border-indigo-400 focus:ring-2 focus:ring-indigo-500/20 dark:focus:ring-indigo-400/20 outline-none resize-none text-gray-800 dark:text-gray-200 placeholder-gray-400 dark:placeholder-gray-500 transition-all duration-200 shadow-lg"
                                style={{ maxHeight: '200px' }}
                            />
                            {input && (
                                <button
                                    onClick={() => setInput('')}
                                    className="absolute right-4 top-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                    </svg>
                                </button>
                            )}
                        </div>
                        <button
                            onClick={() => handleSend()}
                            disabled={!input.trim() || isLoading}
                            className="px-6 py-4 rounded-2xl bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 text-white font-medium shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 hover:scale-105 active:scale-95 flex items-center gap-2"
                        >
                            <span>Send</span>
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                            </svg>
                        </button>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-3 text-center">
                        Press <kbd className="px-2 py-0.5 rounded bg-gray-200 dark:bg-gray-700 font-mono">Enter</kbd> to send, <kbd className="px-2 py-0.5 rounded bg-gray-200 dark:bg-gray-700 font-mono">Shift + Enter</kbd> for new line
                    </p>
                </div>
            </div>
        </div>
    );
}
