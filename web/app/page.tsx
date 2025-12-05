'use client';

import dynamic from 'next/dynamic';

const ChatInterface = dynamic(() => import('./components/ChatInterface'), {
  ssr: false,
  loading: () => (
    <div className="flex flex-col h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-indigo-950 dark:to-purple-950">
      <div className="flex items-center justify-center h-full">
        <div className="flex gap-2">
          <div className="w-3 h-3 rounded-full bg-indigo-500 animate-pulse" style={{ animationDelay: '0s' }}></div>
          <div className="w-3 h-3 rounded-full bg-purple-500 animate-pulse" style={{ animationDelay: '0.2s' }}></div>
          <div className="w-3 h-3 rounded-full bg-pink-500 animate-pulse" style={{ animationDelay: '0.4s' }}></div>
        </div>
      </div>
    </div>
  ),
});

export default function Home() {
  return <ChatInterface />;
}
