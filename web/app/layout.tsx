import type { Metadata } from "next";
// import Script from "next/script";
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
      <body>
        {/* <Script
          src="https://cdn.jsdelivr.net/gh/clashrelated/nepal-relief-banner@1.0.2/banner.min.js"
          integrity="sha384-zsV61Tmn6kfxMnauiuGMOgbTqBC6qhHBy/x0ne2M0sSSoK3q+xTlPs3ZMC8W/SUp"
          crossOrigin="anonymous"
          strategy="afterInteractive"
        /> */}
        {children}
      </body>
    </html>
  );
}
