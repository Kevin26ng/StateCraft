'use client';

import { useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import * as d3 from 'd3';
import { CoalitionGraph } from '@/lib/types';
import { getAgentHex, getAgentName } from '@/lib/agents';

interface CoalitionGraphProps {
  coalitionGraph: CoalitionGraph | null;
}

export default function CoalitionGraphComponent({
  coalitionGraph,
}: CoalitionGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const router = useRouter();
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  useEffect(() => {
    if (!coalitionGraph || !svgRef.current) return;

    const width = 800;
    const height = 400;
    const nodeRadius = 30;

    // Create SVG
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .attr('viewBox', `0 0 ${width} ${height}`)
      .attr('preserveAspectRatio', 'xMidYMid meet')
      .append('g');

    // Create force simulation
    const simulation = d3
      .forceSimulation(coalitionGraph.nodes as any)
      .force(
        'link',
        d3
          .forceLink(coalitionGraph.edges as any)
          .id((d: any) => d.id)
          .distance(100)
          .strength(0.3)
      )
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide(nodeRadius + 10));

    // Draw edges
    const links = g
      .selectAll('line')
      .data(coalitionGraph.edges)
      .enter()
      .append('line')
      .attr('stroke', '#1F2023')
      .attr('stroke-width', (d: any) => Math.max(1, d.weight * 3))
      .attr('opacity', 0.6);

    // Draw nodes
    const nodes = g
      .selectAll('circle')
      .data(coalitionGraph.nodes)
      .enter()
      .append('circle')
      .attr('r', nodeRadius)
      .attr('fill', (d: any) => getAgentHex(d.id))
      .attr('stroke', '#0A0A0A')
      .attr('stroke-width', 2)
      .attr('cursor', 'pointer')
      .attr('opacity', 0.8)
      .on('mouseenter', (e: any, d: any) => {
        setHoveredNode(d.id);
      })
      .on('mouseleave', () => {
        setHoveredNode(null);
      })
      .on('click', (e: any, d: any) => {
        router.push(`?inspect=${d.id}`);
      });

    // Add labels
    const labels = g
      .selectAll('text')
      .data(coalitionGraph.nodes)
      .enter()
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'central')
      .attr('font-size', '10px')
      .attr('fill', '#0A0A0A')
      .attr('font-weight', 'bold')
      .attr('pointer-events', 'none')
      .text((d: any) => coalitionGraph.nodes.indexOf(d));

    // Drag behavior
    const drag = d3
      .drag()
      .on('start', (e: any, d: any) => {
        if (!e.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on('drag', (e: any, d: any) => {
        d.fx = e.x;
        d.fy = e.y;
      })
      .on('end', (e: any, d: any) => {
        if (!e.active) simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    nodes.call(drag as any);

    // Update positions on simulation tick
    simulation.on('tick', () => {
      links
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      nodes
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      labels
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

    // Highlight on hover
    nodes.attr('opacity', (d: any) =>
      !hoveredNode || d.id === hoveredNode ? 0.8 : 0.3
    );

    return () => {
      simulation.stop();
    };
  }, [coalitionGraph, hoveredNode, router]);

  if (!coalitionGraph || coalitionGraph.nodes.length === 0) {
    return (
      <div className="py-12 text-center">
        <p className="text-[#6B7280]">Loading coalition graph...</p>
      </div>
    );
  }

  return (
    <div className="px-12 py-8 border-b border-[#1F2023]">
      <p className="uppercase-label mb-6">Coalition Graph</p>
      <div className="flex justify-center bg-[#111113] p-4 border border-[#1F2023]">
        <svg
          ref={svgRef}
          className="max-w-full"
          style={{ height: 'auto' }}
        />
      </div>
      <p className="text-xs text-[#6B7280] mt-4 text-center">
        Click a node to inspect agent details
      </p>
    </div>
  );
}
