import {useEffect, useMemo, useState} from 'react';

import type {TimelineEntry, TimelineResponse} from '../types/visualization';

type Props={data:TimelineResponse|null;focusedIssueId:string|null};

export default function NarrativeTimeline({data,focusedIssueId}:Props){
  const [kind,setKind]=useState('all');
  const [onlyIssues,setOnlyIssues]=useState(false);
  const [selected,setSelected]=useState<TimelineEntry|null>(null);
  const allEntries=useMemo(()=>data?[...data.groups.flatMap(group=>group.entries),...data.unscheduled]:[],[data]);
  const kinds=useMemo(()=>Array.from(new Set(allEntries.map(entry=>entry.kind))).sort(),[allEntries]);
  const visible=(entry:TimelineEntry)=>(kind==='all'||entry.kind===kind)&&(!onlyIssues||entry.issue_ids.length>0);
  useEffect(()=>setSelected(null),[data]);
  if(!data)return <div className="visualEmpty">完成一次分析后生成时间线。</div>;
  const groups=data.groups.map(group=>({...group,entries:group.entries.filter(visible)})).filter(group=>group.entries.length);
  const unscheduled=data.unscheduled.filter(visible);
  return <div className="timelinePanel">
    <div className="visualControls"><select value={kind} onChange={event=>setKind(event.target.value)}><option value="all">全部记录</option>{kinds.map(value=><option key={value} value={value}>{value}</option>)}</select><label><input type="checkbox" checked={onlyIssues} onChange={event=>setOnlyIssues(event.target.checked)}/>只看问题记录</label><span>{groups.reduce((sum,group)=>sum+group.entries.length,0)+unscheduled.length} 条</span></div>
    {data.warnings.map(warning=><p className="visualWarning" key={warning}>{warning}</p>)}
    <div className="timelineList">{groups.map(group=><section key={group.sort_key}><time>{group.timestamp}</time><div>{group.entries.map(entry=><button key={entry.id} className={`${entry.issue_ids.length?'hasIssue ':''}${focusedIssueId&&entry.issue_ids.includes(focusedIssueId)?'focused':''}`} onClick={()=>setSelected(entry)}><b>{entry.title}</b><small>{entry.kind} · {entry.evidence.document_name}:{entry.evidence.line_start}</small></button>)}</div></section>)}
      {unscheduled.length>0&&<section className="unscheduled"><time>时间未确定</time><div>{unscheduled.map(entry=><button key={entry.id} className={`${entry.issue_ids.length?'hasIssue ':''}${focusedIssueId&&entry.issue_ids.includes(focusedIssueId)?'focused':''}`} onClick={()=>setSelected(entry)}><b>{entry.title}</b><small>{entry.timestamp||'无时间'} · {entry.evidence.document_name}:{entry.evidence.line_start}</small></button>)}</div></section>}
    </div>
    {!groups.length&&!unscheduled.length&&<div className="visualEmpty">当前筛选下没有时间记录。</div>}
    {selected&&<div className="visualDetail"><b>{selected.title}</b><span>{selected.timestamp||'未标注时间'} · {selected.evidence.document_name}:{selected.evidence.line_start}</span><p>{selected.evidence.text}</p>{selected.issue_ids.length>0&&<small>关联 {selected.issue_ids.length} 个一致性问题</small>}</div>}
  </div>;
}
