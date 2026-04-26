'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { HistoryRun } from '@/lib/types';

interface AggregateChartGridProps {
  runs: HistoryRun[];
}

export default function AggregateChartGrid({ runs }: AggregateChartGridProps) {
  if (runs.length === 0) {
    return (
      <div className="py-12 text-center">
        <p className="text-[#9CA3AF]">No data available for charts.</p>
      </div>
    );
  }

  // Prepare data for charts - use run index as pseudo turn number
  const chartData = runs.map((run, index) => ({
    turn: index,
    gdp: run.society_score * 0.8, // Simulated metric
    inflation: 50 - run.society_score * 0.3,
    resources: run.society_score * 0.7,
    stability: run.society_score * 0.9,
    mortality: 100 - run.society_score,
    public_trust: run.society_score * 0.85,
  }));

  const chartConfig = [
    {
      title: 'GDP',
      dataKey: 'gdp',
      stroke: '#7DD3FC',
    },
    {
      title: 'Inflation',
      dataKey: 'inflation',
      stroke: '#7DD3FC',
    },
    {
      title: 'Resources',
      dataKey: 'resources',
      stroke: '#7DD3FC',
    },
    {
      title: 'Stability',
      dataKey: 'stability',
      stroke: '#7DD3FC',
    },
    {
      title: 'Mortality',
      dataKey: 'mortality',
      stroke: '#7DD3FC',
    },
    {
      title: 'Public Trust',
      dataKey: 'public_trust',
      stroke: '#7DD3FC',
    },
  ];

  const ChartWrapper = ({ children, title }: { children: React.ReactNode; title: string }) => (
    <div className="border border-[#1F2023] p-4 bg-[#111113]">
      <p className="uppercase-label mb-4">{title}</p>
      {children}
    </div>
  );

  return (
    <div className="grid grid-cols-2 gap-4">
      {chartConfig.map((config) => (
        <ChartWrapper key={config.dataKey} title={config.title}>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1F2023" />
              <XAxis
                dataKey="turn"
                tick={{ fill: '#6B7280', fontSize: 11 }}
                stroke="#1F2023"
              />
              <YAxis
                tick={{ fill: '#6B7280', fontSize: 11 }}
                stroke="#1F2023"
                domain={[0, 100]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#111113',
                  border: '1px solid #1F2023',
                  borderRadius: '0px',
                }}
                labelStyle={{ color: '#F3F4F6' }}
                formatter={(value) => {
                  if (typeof value === 'number') {
                    return value.toFixed(1);
                  }
                  return value;
                }}
              />
              <Line
                type="monotone"
                dataKey={config.dataKey}
                stroke={config.stroke}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartWrapper>
      ))}
    </div>
  );
}
