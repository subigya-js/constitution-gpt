import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Constitution GPT - AI-Powered Constitutional Intelligence",
  description: "Ask any question about the Constitution of Nepal. Get accurate, citation-backed answers with proper hierarchical structure powered by RAG and LLMs.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
