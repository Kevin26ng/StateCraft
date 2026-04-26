'use client';

export function SkeletonBox({ className = '' }: { className?: string }) {
  return (
    <div
      className={`bg-[#111113] animate-pulse ${className}`}
    />
  );
}

export function SkeletonText({ width = 'w-32', height = 'h-4' }: { width?: string; height?: string }) {
  return <div className={`${width} ${height} bg-[#111113] rounded-sm animate-pulse`} />;
}

export function SkeletonCard() {
  return (
    <div className="border border-[#1F2023] p-4">
      <SkeletonText width="w-24" height="h-4" />
      <div className="mt-3 space-y-2">
        <SkeletonText width="w-full" height="h-3" />
        <SkeletonText width="w-3/4" height="h-3" />
      </div>
    </div>
  );
}

export function SkeletonTableRow() {
  return (
    <tr className="border-b border-[#1F2023]">
      <td className="px-4 py-4"><SkeletonText width="w-16" /></td>
      <td className="px-4 py-4"><SkeletonText width="w-24" /></td>
      <td className="px-4 py-4"><SkeletonText width="w-12" /></td>
      <td className="px-4 py-4"><SkeletonText width="w-12" /></td>
      <td className="px-4 py-4"><SkeletonText width="w-12" /></td>
      <td className="px-4 py-4"><SkeletonText width="w-20" /></td>
    </tr>
  );
}

export function LoadingSpinner() {
  return (
    <div className="flex items-center justify-center py-8">
      <div
        className="w-8 h-8 border-2 border-[#1F2023] border-t-[#7DD3FC] rounded-full"
        style={{
          animation: 'spin 1s linear infinite',
        }}
      />
    </div>
  );
}
