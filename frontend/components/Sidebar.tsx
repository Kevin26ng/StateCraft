'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useStore } from '@/lib/store';

export default function Sidebar() {
  const pathname = usePathname();
  const currentRunId = useStore((state) => state.currentRunId);

  const activeRunHref = currentRunId ? `/run/${currentRunId}` : null;
  const isOnRun = pathname.startsWith('/run/');

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  return (
    <aside className="w-[200px] border-r border-[#1F2023] bg-[#0A0A0A] sticky top-0 h-screen flex flex-col">
      {/* Logo */}
      <div className="px-5 py-7 border-b border-[#1F2023]">
        <h1 className="text-[#F3F4F6] text-sm font-semibold tracking-[0.15em] uppercase">
          StateCraft
        </h1>
      </div>

      {/* Navigation */}
      <nav className="flex-1 pt-6">
        <p className="px-5 mb-4 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#4B5563]">
          Simulation
        </p>
        <ul className="space-y-0.5">
          {/* Mission Control */}
          <li>
            <Link
              href="/"
              className={`flex items-center gap-3 px-5 py-2.5 text-sm border-l-2 transition-all duration-200 group ${
                pathname === '/'
                  ? 'border-[#7DD3FC] text-[#7DD3FC] font-medium bg-[#7DD3FC]/5'
                  : 'border-transparent text-[#9CA3AF] hover:text-[#F3F4F6] hover:bg-white/[0.02]'
              }`}
            >
              <span className={`transition-colors ${pathname === '/' ? 'text-[#7DD3FC]' : 'text-[#6B7280] group-hover:text-[#9CA3AF]'}`}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
                  <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
                </svg>
              </span>
              Mission Control
            </Link>
          </li>

          {/* Active Run — only navigable when a run exists */}
          <li>
            {activeRunHref ? (
              <Link
                href={activeRunHref}
                className={`flex items-center gap-3 px-5 py-2.5 text-sm border-l-2 transition-all duration-200 group ${
                  isOnRun
                    ? 'border-[#7DD3FC] text-[#7DD3FC] font-medium bg-[#7DD3FC]/5'
                    : 'border-transparent text-[#9CA3AF] hover:text-[#F3F4F6] hover:bg-white/[0.02]'
                }`}
              >
                <span className={`transition-colors ${isOnRun ? 'text-[#7DD3FC]' : 'text-[#6B7280] group-hover:text-[#9CA3AF]'}`}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                </span>
                Active Run
                {isOnRun && (
                  <span className="ml-auto w-1.5 h-1.5 rounded-full bg-[#34D399] animate-pulse" />
                )}
              </Link>
            ) : (
              <span className="flex items-center gap-3 px-5 py-2.5 text-sm border-l-2 border-transparent text-[#4B5563] cursor-default select-none">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                Active Run
              </span>
            )}
          </li>

          {/* History */}
          <li>
            <Link
              href="/history"
              className={`flex items-center gap-3 px-5 py-2.5 text-sm border-l-2 transition-all duration-200 group ${
                isActive('/history')
                  ? 'border-[#7DD3FC] text-[#7DD3FC] font-medium bg-[#7DD3FC]/5'
                  : 'border-transparent text-[#9CA3AF] hover:text-[#F3F4F6] hover:bg-white/[0.02]'
              }`}
            >
              <span className={`transition-colors ${isActive('/history') ? 'text-[#7DD3FC]' : 'text-[#6B7280] group-hover:text-[#9CA3AF]'}`}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" />
                </svg>
              </span>
              History
            </Link>
          </li>
        </ul>
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-[#1F2023]">
        <p className="text-[10px] text-[#4B5563]">v0.1.0 · API :5000</p>
      </div>
    </aside>
  );
}
