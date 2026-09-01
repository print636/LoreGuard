import {useEffect, useMemo, useRef, useState} from 'react';
import cytoscape, {type ElementDefinition} from 'cytoscape';

import type {GraphEdge, GraphResponse} from '../types/visualization';

type Props={data:GraphResponse|null;focusedIssueId:string|null};

export default function RelationGraph({data,focusedIssueId}:Props){
  const containerRef=useRef<HTMLDivElement|null>(null);
  const [kind,setKind]=useState('all');
  const [onlyIssues,setOnlyIssues]=useState(false);
  const [selected,setSelected]=useState<GraphEdge|null>(null);
  const kinds=useMemo(()=>Array.from(new Set(data?.edges.map(edge=>edge.type)||[])).sort(),[data]);
  const edges=useMemo(()=>data?.edges.filter(edge=>(kind==='all'||edge.type===kind)&&(!onlyIssues||edge.issue_ids.length>0))||[],[data,kind,onlyIssues]);

  useEffect(()=>{
    if(!containerRef.current||!data)return;
    const nodeIds=new Set(edges.flatMap(edge=>[edge.source,edge.target]));
    const nodes=data.nodes.filter(node=>nodeIds.has(node.id));
    const elements:ElementDefinition[]=[
      ...nodes.map(node=>({data:{id:node.id,label:node.label,nodeType:node.type,hasIssue:node.issue_ids.length>0,focused:Boolean(focusedIssueId&&node.issue_ids.includes(focusedIssueId))}})),
      ...edges.map(edge=>({data:{id:edge.id,source:edge.source,target:edge.target,label:edge.label,edgeType:edge.type,hasIssue:edge.issue_ids.length>0,focused:Boolean(focusedIssueId&&edge.issue_ids.includes(focusedIssueId))}})),
    ];
    const cy=cytoscape({
      container:containerRef.current,
      elements,
      layout:{name:'cose',animate:false,padding:24,nodeRepulsion:6200,idealEdgeLength:115},
      style:[
        {selector:'node',style:{'background-color':'#295c49','border-color':'#74df9f','border-width':1,'color':'#dfeae6','font-size':11,'label':'data(label)','text-wrap':'wrap','text-max-width':'100px','text-valign':'bottom','text-margin-y':7,'width':34,'height':34}},
        {selector:'node[hasIssue = true]',style:{'border-color':'#f0a27f','border-width':3}},
        {selector:'edge',style:{'curve-style':'bezier','line-color':'#5f8d7a','target-arrow-color':'#5f8d7a','target-arrow-shape':'triangle','width':1.5,'label':'data(label)','color':'#a9bbb4','font-size':9,'text-background-color':'#07110e','text-background-opacity':.84,'text-background-padding':'2px'}},
        {selector:'edge[hasIssue = true]',style:{'line-color':'#d9816c','target-arrow-color':'#d9816c','width':3}},
        {selector:'[focused = true]',style:{'border-color':'#f4dc78','line-color':'#f4dc78','target-arrow-color':'#f4dc78','z-index':20}},
      ],
    });
    cy.on('tap','edge',event=>setSelected(data.edges.find(edge=>edge.id===event.target.id())||null));
    cy.on('tap','node',event=>{
      const first=event.target.connectedEdges().first();
      setSelected(first.nonempty()?data.edges.find(edge=>edge.id===first.id())||null:null);
    });
    return()=>cy.destroy();
  },[data,edges,focusedIssueId]);

  useEffect(()=>{if(selected&&!edges.some(edge=>edge.id===selected.id))setSelected(null)},[edges,selected]);
  if(!data)return <div className="visualEmpty">完成一次分析后生成关系图。</div>;
  return <div className="graphPanel">
    <div className="visualControls"><select value={kind} onChange={event=>setKind(event.target.value)}><option value="all">全部关系</option>{kinds.map(value=><option key={value} value={value}>{value}</option>)}</select><label><input type="checkbox" checked={onlyIssues} onChange={event=>setOnlyIssues(event.target.checked)}/>只看问题关系</label><span>{edges.length} 条边 · {data.nodes.length} 个总节点</span></div>
    {data.warnings.map(warning=><p className="visualWarning" key={warning}>{warning}</p>)}
    {edges.length?<div ref={containerRef} className="graphCanvas"/>:<div className="visualEmpty">当前筛选下没有关系。</div>}
    {selected&&<div className="visualDetail"><b>{selected.label} · {selected.type}</b><span>{selected.timestamp||'未标注时间'} · {selected.evidence.document_name}:{selected.evidence.line_start}</span><p>{selected.evidence.text}</p>{selected.issue_ids.length>0&&<small>关联 {selected.issue_ids.length} 个一致性问题</small>}</div>}
  </div>;
}
