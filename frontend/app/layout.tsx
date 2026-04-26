import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import Sidebar from '@/components/Sidebar';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: 'StateCraft',
  description: 'Multi-agent governance training console — train six AI agents under crisis scenarios',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} antialiased`}>
        <div className="flex h-screen bg-[#0A0A0A] text-[#F3F4F6]">
          <Sidebar />
          <main className="flex-1 overflow-auto relative">
            {/* Subtle grid background */}
            <div className="absolute inset-0 bg-grid-pattern pointer-events-none opacity-40" />
            <div className="relative z-10">
              {children}
            </div>
          </main>
        </div>
      </body>
    </html>
  );
}
