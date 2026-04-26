'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    { label: 'Mission Control', href: '/' },
    { label: 'Active Run', href: '/run' },
    { label: 'History', href: '/history' },
  ];

  const isActive = (href: string): boolean => {
    if (href === '/') {
      return pathname === '/';
    }
    return pathname.startsWith(href);
  };

  return (
    <aside className="w-[200px] border-r border-[#1F2023] bg-[#0A0A0A] sticky top-0 h-screen flex flex-col">
      {/* Logo/Header */}
      <div className="px-6 py-8 border-b border-[#1F2023]">
        <h1 className="text-[#F3F4F6] text-base font-normal tracking-wide">
          STATECRAFT
        </h1>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-8">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const active = isActive(item.href);
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  className={`block px-6 py-3 text-sm border-l-2 transition-colors ${
                    active
                      ? 'border-[#7DD3FC] text-[#7DD3FC] font-medium'
                      : 'border-transparent text-[#9CA3AF] hover:text-[#F3F4F6]'
                  }`}
                >
                  {item.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer spacer */}
      <div className="h-8"></div>
    </aside>
  );
}
