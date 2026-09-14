'use client';

import { useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

type MessageStatus = 'complete' | 'waiting' | 'streaming' | 'error';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  status: MessageStatus;
  sources?: Source[];
  statusStep?: string;
  statusLabel?: string;
}

interface Source {
  title: string;
  url: string;
  source_type: string;
}

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000').replace(/\/$/, '');

const STARTERS = [
  { eyebrow: 'Federal executive', question: 'How is the Prime Minister elected in Nepal?' },
  { eyebrow: 'Article 91', question: 'What does Article 91 mention?' },
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

/** Strip trailing inline citation links injected by the research API, e.g. ([domain.com](https://...)) */
function stripInlineCitations(text: string): string {
  return text
    .replace(/\s*\(\[[^\]]+\]\(https?:\/\/[^)]+\)\)/g, '')
    .trim();
}

function AnswerContent({ content }: { content: string }) {
  return (
    <div className="answer-content">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Open all links in a new tab safely
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer noopener">{children}</a>
          ),
        }}
      >
        {stripInlineCitations(content)}
      </ReactMarkdown>
    </div>
  );
}

const STEP_META: Record<string, { color: string }> = {
  resolving:   { color: '#7c9ef5' },
  classifying: { color: '#a78bfa' },
  retrieving:  { color: '#34d399' },
  researching: { color: '#fb923c' },
  composing:   { color: '#f472b6' },
};

function StatusIndicator({ step, label }: { step?: string; label?: string }) {
  const meta = step ? STEP_META[step] : null;
  const displayLabel = label ?? 'Thinking\u2026';
  return (
    <div className="status-indicator" role="status" aria-live="polite">
      <div className="status-top-row">
        <span
          className="status-step-dot"
          key={step ?? 'idle'}
          style={meta ? { background: meta.color } : { background: '#555' }}
        />
        <span className="status-label-text" key={displayLabel}>{displayLabel}</span>
      </div>
      <div className="thinking">
        <span /><span /><span />
      </div>
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
  const conversationIdRef = useRef<string | null>(null);

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

  const revealResponse = async (id: string, answer: string, sources: Source[], signal: AbortSignal) => {
    const chunks = answer.match(/\S+\s*/g) ?? [answer];
    let visible = '';
    updateAssistant(id, { status: 'streaming' });

    for (let index = 0; index < chunks.length; index += 3) {
      visible += chunks.slice(index, index + 3).join('');
      updateAssistant(id, { content: visible });
      await wait(index < 12 ? 36 : 22, signal);
    }
    updateAssistant(id, { content: answer, sources, status: 'complete' });
  };

  const handleSend = async (suggestedQuestion?: string) => {
    const question = (suggestedQuestion ?? input).trim();
    if (!question || isGenerating) return;

    const responseId = crypto.randomUUID();
    const conversationId = conversationIdRef.current ?? crypto.randomUUID();
    conversationIdRef.current = conversationId;
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
      const response = await fetch(`${API_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, conversation_id: conversationId }),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
      if (!response.body) throw new Error('No response body from stream endpoint');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let finalAnswer: string | null = null;
      let finalSources: Source[] = [];

      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          let event: Record<string, unknown>;
          try { event = JSON.parse(line.slice(6)); } catch { continue; }

          if (event.type === 'status') {
            updateAssistant(responseId, {
              statusStep: event.step as string,
              statusLabel: event.label as string,
            });
          } else if (event.type === 'result') {
            if (event.conversation_id) conversationIdRef.current = event.conversation_id as string;
            finalAnswer = event.answer as string;
            finalSources = (event.sources ?? []) as Source[];
            break outer;
          } else if (event.type === 'error') {
            throw new Error((event.detail as string) ?? 'Stream error');
          }
        }
      }

      if (!finalAnswer?.trim()) throw new Error('The API returned an empty answer');
      await revealResponse(responseId, finalAnswer, finalSources, controller.signal);
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
    conversationIdRef.current = null;
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
                        <StatusIndicator step={message.statusStep} label={message.statusLabel} />
                      ) : (
                        <>
                          <AnswerContent content={message.content} />
                          {message.status !== 'streaming' && message.sources && message.sources.length > 0 && (
                            <div className="source-list" aria-label="Research sources">
                              <h4>Sources</h4>
                              {message.sources.map((source, index) => (
                                <a href={source.url} target="_blank" rel="noreferrer" key={source.url}>
                                  <span>{index + 1}</span>{source.title}
                                </a>
                              ))}
                            </div>
                          )}
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
            <textarea ref={textareaRef} value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={handleKeyDown} placeholder="Ask about Nepal’s Constitution or law" rows={1} aria-label="Message Constitution GPT" />
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
