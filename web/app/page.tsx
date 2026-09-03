'use client';

import dynamic from 'next/dynamic';

const ChatInterface = dynamic(() => import('./components/ChatInterface'), {
  ssr: false,
  loading: () => (
    <div className="h-screen bg-[#212121] text-[#ececec] grid place-items-center">
      <div className="flex gap-1.5" aria-label="Loading">
        <span className="w-1.5 h-1.5 rounded-full bg-[#a9a9a9] animate-pulse" />
        <span className="w-1.5 h-1.5 rounded-full bg-[#a9a9a9] animate-pulse [animation-delay:150ms]" />
        <span className="w-1.5 h-1.5 rounded-full bg-[#a9a9a9] animate-pulse [animation-delay:300ms]" />
      </div>
    </div>
  ),
});

export default function Home() {
  return <ChatInterface />;
}
