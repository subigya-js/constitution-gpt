'use client';

import { useEffect, useRef, useState } from 'react';

type MessageStatus = 'complete' | 'waiting' | 'streaming' | 'error';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  status: MessageStatus;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

const STARTERS = [
  { eyebrow: 'Federal executive', question: 'How is the Prime Minister elected in Nepal?' },
  { eyebrow: 'Fundamental rights', question: 'What rights does the Constitution guarantee?' },
  { eyebrow: 'Civic responsibility', question: 'What are the constitutional duties of citizens?' },
  { eyebrow: 'Federal parliament', question: 'How is Nepal’s Federal Parliament structured?' },
];

const wait = (milliseconds: number, signal: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = window.setTimeout(resolve, milliseconds);
    signal.addEventListener('abort', () => {
      window.clearTimeout(timer);
      reject(new DOMException('Stopped', 'AbortError'));
    }, { once: true });
  });

function AnswerContent({ content }: { content: string }) {
  return (
    <div className="answer-content">
      {content.split('\n').map((line, index) => {
        const trimmed = line.trim();
        if (!trimmed) return <div className="answer-spacer" key={index} />;
        if (/^(#{1,3}\s|📘|Article\s+\d+)/i.test(trimmed)) {
          return <h3 key={index}>{trimmed.replace(/^#{1,3}\s*/, '')}</h3>;
        }
        if (/^([•*-]|\d+\.)\s/.test(trimmed)) {
          return <p className="answer-list" key={index}>{trimmed}</p>;
        }
        return <p key={index}>{line}</p>;
      })}
    </div>
  );
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const transcriptRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const transcript = transcriptRef.current;
    transcript?.scrollTo({ top: transcript.scrollHeight, behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = '0px';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 168)}px`;
  }, [input]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const updateAssistant = (id: string, patch: Partial<Message>) => {
    setMessages((current) => current.map((message) =>
      message.id === id ? { ...message, ...patch } : message,
    ));
  };

  const revealResponse = async (id: string, answer: string, signal: AbortSignal) => {
    const chunks = answer.match(/\S+\s*/g) ?? [answer];
    let visible = '';
    updateAssistant(id, { status: 'streaming' });

    for (let index = 0; index < chunks.length; index += 3) {
      visible += chunks.slice(index, index + 3).join('');
      updateAssistant(id, { content: visible });
      await wait(index < 12 ? 36 : 22, signal);
    }
    updateAssistant(id, { content: answer, status: 'complete' });
  };

  const handleSend = async (suggestedQuestion?: string) => {
    const question = (suggestedQuestion ?? input).trim();
    if (!question || isGenerating) return;

    const responseId = crypto.randomUUID();
    const controller = new AbortController();
    abortRef.current = controller;
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: 'user', content: question, status: 'complete' },
      { id: responseId, role: 'assistant', content: '', status: 'waiting' },
    ]);
    setInput('');
    setIsGenerating(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);

      const data: { answer?: string } = await response.json();
      if (!data.answer?.trim()) throw new Error('The API returned an empty answer');
      await revealResponse(responseId, data.answer, controller.signal);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        setMessages((current) => current.flatMap((message) => {
          if (message.id !== responseId) return [message];
          return message.content ? [{ ...message, status: 'complete' }] : [];
        }));
      } else {
        updateAssistant(responseId, {
          status: 'error',
          content: 'I couldn’t reach the Constitution GPT service. Make sure the API is running, then try again.',
        });
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  };

  const startNewChat = () => {
    abortRef.current?.abort();
    setMessages([]);
    setInput('');
    window.setTimeout(() => textareaRef.current?.focus(), 0);
  };

  const copyAnswer = async (message: Message) => {
    await navigator.clipboard.writeText(message.content);
    setCopiedId(message.id);
    window.setTimeout(() => setCopiedId(null), 1600);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      void handleSend();
    }
  };

  return (
    <div className="chat-shell">
      <main className="chat-main">
        <header className="topbar">
          <span className="model-name">Constitution GPT</span>
          {messages.length > 0 && (
            <button className="header-new-chat cursor-pointer" aria-label="Start a new chat" onClick={startNewChat}>New chat</button>
          )}
        </header>

        <div className="transcript" ref={transcriptRef}>
          {messages.length === 0 ? (
            <section className="empty-state">
              <h2>What would you like to understand?</h2>
              <p>Explore Nepal’s Constitution with clear, source-grounded answers.</p>
              <div className="starter-grid">
                {STARTERS.map((starter, index) => (
                  <button key={starter.question} className="starter-card" style={{ '--delay': `${index * 70}ms` } as React.CSSProperties} onClick={() => void handleSend(starter.question)}>
                    <span>{starter.eyebrow}</span><strong>{starter.question}</strong><i aria-hidden="true">↗</i>
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article className={`message-row ${message.role}`} key={message.id}>
                  <div className="message-body">
                    {message.role === 'user' ? <div className="user-bubble">{message.content}</div>
                      : message.status === 'waiting' ? (
                        <div className="thinking" role="status"><span /><span /><span /><em>Reading the Constitution</em></div>
                      ) : (
                        <>
                          <AnswerContent content={message.content} />
                          {message.status === 'streaming' && <span className="stream-caret" aria-label="Response is streaming" />}
                          {message.status !== 'streaming' && message.content && (
                            <div className="message-actions"><button onClick={() => void copyAnswer(message)} aria-label="Copy answer">{copiedId === message.id ? 'Copied' : 'Copy'}</button></div>
                          )}
                        </>
                      )}
                  </div>
                </article>
              ))}
            </div>
          )}
        </div>

        <footer className="composer-zone">
          {isGenerating && <button className="stop-button" onClick={() => abortRef.current?.abort()}><span aria-hidden="true" /> Stop generating</button>}
          <div className="composer">
            <textarea ref={textareaRef} value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={handleKeyDown} placeholder="Ask about Nepal’s Constitution" rows={1} aria-label="Message Constitution GPT" />
            <div className="composer-controls">
              {isGenerating ? (
                <button className="send-button stop-inline-button" aria-label="Stop generating" onClick={() => abortRef.current?.abort()}>
                  <span aria-hidden="true" />
                </button>
              ) : (
                <button className="send-button" aria-label="Send message" disabled={!input.trim()} onClick={() => void handleSend()}>↑</button>
              )}
            </div>
          </div>
          <p className="disclaimer">Constitution GPT can make mistakes. Verify important legal information with an official source.</p>
        </footer>
      </main>
    </div>
  );
}
