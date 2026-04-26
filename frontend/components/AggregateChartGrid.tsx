'use client';

import { useMemo } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { HistoryRun } from '@/lib/types';

interface AggregateChartGridProps {
  runs: HistoryRun[];
}

const C = {
  bg: '#111113',
  grid: '#1F2023',
  text: '#6B7280',
  accent: '#7DD3FC',
  green: '#34D399',
  amber: '#F59E0B',
  red: '#EF4444',
};

function Tip({ active, payload, label }: { active?: boolean; payload?: Array<{ color?: string; name: string; value: number }>; label?: string | number }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#18181B] border border-[#2A2A2E] px-3 py-2 text-xs">
      {label != null && <p className="text-[#6B7280] mb-1">{label}</p>}
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color ?? C.accent }}>
          {p.name}: {typeof p.value === 'number' ? (Number.isInteger(p.value) ? p.value : p.value.toFixed(1)) : p.value}
        </p>
      ))}
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-[#111113] border border-[#1F2023] p-4">
      <p className="text-[10px] uppercase tracking-wider text-[#4B5563] mb-3">{title}</p>
      {children}
    </div>
  );
}

export default function AggregateChartGrid({ runs }: AggregateChartGridProps) {
  const scoreData = useMemo(
    () => runs.map((r, i) => ({ run: i + 1, score: Math.round(r.society_score) })),
    [runs]
  );

  const turnsData = useMemo(
    () => runs.map((r, i) => ({ run: i + 1, turns: r.turns_survived })),
    [runs]
  );

  const scenarioData = useMemo(() => {
    const map: Record<string, { count: number; total: number }> = {};
    runs.forEach((r) => {
      if (!map[r.scenario]) map[r.scenario] = { count: 0, total: 0 };
      map[r.scenario].count++;
      map[r.scenario].total += r.society_score;
    });
    return Object.entries(map).map(([scenario, { count, total }]) => ({
      scenario,
      avgScore: Math.round(total / count),
    }));
  }, [runs]);

  const difficultyData = useMemo(() => {
    const map: Record<number, number> = {};
    runs.forEach((r) => { map[r.difficulty_tier] = (map[r.difficulty_tier] ?? 0) + 1; });
    return Object.entries(map)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([tier, count]) => ({ tier: `T${tier}`, count }));
  }, [runs]);

  const outcomeData = useMemo(() => [
    { name: 'Completed', count: runs.filter(r => (r.status ?? 'completed') === 'completed').length },
    { name: 'Collapsed', count: runs.filter(r => r.status === 'collapsed').length },
  ], [runs]);

  const trendData = useMemo(() => {
    const W = 5;
    return runs.map((r, i) => {
      const slice = runs.slice(Math.max(0, i - W + 1), i + 1);
      return {
        run: i + 1,
        avg: Math.round(slice.reduce((s, x) => s + x.society_score, 0) / slice.length),
      };
    });
  }, [runs]);

  const axisTick = { fill: C.text, fontSize: 10 };
  const axisLine = false as const;
  const tickLine = false as const;

  return (
    <div className="grid grid-cols-3 gap-4">
      <Panel title="Society Score">
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={scoreData}>
            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="run" tick={axisTick} tickLine={tickLine} axisLine={axisLine} />
            <YAxis tick={axisTick} tickLine={tickLine} axisLine={axisLine} domain={[0, 100]} />
            <Tooltip content={<Tip />} />
            <Line type="monotone" dataKey="score" stroke={C.accent} strokeWidth={1.5} dot={false} name="Score" />
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Turns Survived">
        <ResponsiveContainer width="100%" height={150}>
          <BarChart data={turnsData} barSize={6}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="run" tick={axisTick} tickLine={tickLine} axisLine={axisLine} />
            <YAxis tick={axisTick} tickLine={tickLine} axisLine={axisLine} domain={[0, 30]} />
            <Tooltip content={<Tip />} />
            <Bar dataKey="turns" fill={C.accent} name="Turns" opacity={0.8} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Score by Scenario">
        <ResponsiveContainer width="100%" height={150}>
          <BarChart data={scenarioData} barSize={20}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="scenario" tick={axisTick} tickLine={tickLine} axisLine={axisLine} />
            <YAxis tick={axisTick} tickLine={tickLine} axisLine={axisLine} domain={[0, 100]} />
            <Tooltip content={<Tip />} />
            <Bar dataKey="avgScore" fill={C.green} name="Avg Score" opacity={0.8} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Difficulty Distribution">
        <ResponsiveContainer width="100%" height={150}>
          <BarChart data={difficultyData} barSize={28}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="tier" tick={axisTick} tickLine={tickLine} axisLine={axisLine} />
            <YAxis tick={axisTick} tickLine={tickLine} axisLine={axisLine} allowDecimals={false} />
            <Tooltip content={<Tip />} />
            <Bar dataKey="count" fill={C.amber} name="Count" opacity={0.8} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Outcomes">
        <ResponsiveContainer width="100%" height={150}>
          <BarChart data={outcomeData} barSize={36}>
            <CartesianGrid stroke={C.grid} vertical={false} />
            <XAxis dataKey="name" tick={axisTick} tickLine={tickLine} axisLine={axisLine} />
            <YAxis tick={axisTick} tickLine={tickLine} axisLine={axisLine} allowDecimals={false} />
            <Tooltip content={<Tip />} />
            <Bar dataKey="count" name="Count" opacity={0.8}>
              <Cell fill={C.green} />
              <Cell fill={C.red} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Score Trend (5-run avg)">
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={trendData}>
            <CartesianGrid stroke={C.grid} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="run" tick={axisTick} tickLine={tickLine} axisLine={axisLine} />
            <YAxis tick={axisTick} tickLine={tickLine} axisLine={axisLine} domain={[0, 100]} />
            <Tooltip content={<Tip />} />
            <Line type="monotone" dataKey="avg" stroke={C.green} strokeWidth={1.5} dot={false} name="Avg" />
          </LineChart>
        </ResponsiveContainer>
      </Panel>
    </div>
  );
}
