export type Evidence = {
  document_id: string;
  document_name: string;
  line_start: number;
  line_end: number;
  text: string;
};

export type GraphNode = {
  id: string;
  label: string;
  type: string;
  issue_ids: string[];
  metadata: Record<string, unknown>;
};

export type GraphEdge = {
  id: string;
  source: string;
  target: string;
  type: string;
  label: string;
  record_id: string;
  timestamp: string | null;
  evidence: Evidence;
  issue_ids: string[];
  metadata: Record<string, string>;
};

export type GraphResponse = {
  run_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
  warnings: string[];
};

export type TimelineEntry = {
  id: string;
  record_id: string;
  kind: string;
  title: string;
  timestamp: string | null;
  precision: 'exact' | 'date' | 'relative' | 'unknown';
  evidence: Evidence;
  issue_ids: string[];
  attrs: Record<string, string>;
};

export type TimelineGroup = {
  timestamp: string;
  sort_key: string;
  precision: string;
  entries: TimelineEntry[];
};

export type TimelineResponse = {
  run_id: string;
  groups: TimelineGroup[];
  unscheduled: TimelineEntry[];
  warnings: string[];
};
