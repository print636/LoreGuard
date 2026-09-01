import {lazy, Suspense, useEffect, useMemo, useRef, useState} from 'react';
import NarrativeTimeline from './components/NarrativeTimeline';
import type {Evidence,GraphResponse,TimelineResponse} from './types/visualization';

const RelationGraph=lazy(()=>import('./components/RelationGraph'));

const API = import.meta.env.VITE_API_BASE || '';
type FeedbackState={id:string;label:string;comment:string;created_at:string};
type Issue={id:string;category:string;severity:string;confidence:number;title:string;explanation:string;suggestion:string;evidence:Evidence[]};
type RecordRow={id:string;kind:string;attrs:Record<string,string>;evidence:Evidence};
type Doc={id:string;project_id:string;name:string;version:number;active:boolean;created_at:string;content:string};
type DiffLine={type:'added'|'removed'|'unchanged';content:string;old_line:number|null;new_line:number|null};
type DocumentDiff={from_document:Doc&{char_count:number;line_count:number};to_document:Doc&{char_count:number;line_count:number};summary:{added_lines:number;removed_lines:number;unchanged_lines:number;changed_hunks:number;compared_old_lines:number;compared_new_lines:number;old_total_lines:number;new_total_lines:number;input_truncated:boolean;output_truncated:boolean};hunks:Array<{old_start:number;old_lines:number;new_start:number;new_lines:number;lines:DiffLine[]}>;warnings:string[]};
type RunInfo={id:string;project_id:string;prompt_tokens:number;completion_tokens:number;estimated_cost_usd:number|null;status:string;error?:string|null;created_at:string};
type Project={id:string;name:string;description:string;created_at:string;active_document_count:number;latest_run:RunInfo|null};
type Diagnostics={chunking?:{total_chunks:number;documents:Array<{document_name:string;chunk_count:number;chars:number;would_truncate_model_chunks:boolean}>};aliases?:{declaration_count:number;trace_count:number;traces:Array<Record<string,unknown>>};retrieval?:{candidate_count:number;consumed_count:number;traces:Array<Record<string,unknown>>;boundary?:string};timings?:{chunk_ms:number;extract_ms:number;index_ms:number;check_ms:number;report_ms:number;total_ms:number;first_progress_ms:number}};

const defaultWorld=`# 世界观设定

林澈的发色是银色。
星门只能由潮汐晶核驱动。
1026-04-03 08:00，苏弦保管星门钥匙。
1026-04-03 12:00，林澈得知星门口令。`;
const defaultChapter=`# 第一章

林澈的发色是黑色。
1026-04-03 10:00，林澈在北港。
1026-04-03 10:00，林澈在南塔。
1026-04-03 09:00，林澈说出星门口令。
1026-04-03 09:30，林澈使用星门钥匙。
星门由普通火焰驱动。`;
const categoryNames:Record<string,string>={fact_conflict:'事实冲突',location_collision:'同刻多地点',knowledge_without_acquisition:'知识越权',item_ownership:'物品状态',world_rule_conflict:'世界规则'};
const feedbackNames:Record<string,string>={accepted:'已接受',false_positive:'误报',resolved:'已解决'};

export default function App(){
  const [projects,setProjects]=useState<Project[]>([]); const [project,setProject]=useState(''); const [projectName,setProjectName]=useState('');
  const [docs,setDocs]=useState<Doc[]>([]); const [runs,setRuns]=useState<RunInfo[]>([]); const [replaceId,setReplaceId]=useState(''); const [files,setFiles]=useState<File[]>([]);
  const [diffFrom,setDiffFrom]=useState(''); const [diffTo,setDiffTo]=useState(''); const [documentDiff,setDocumentDiff]=useState<DocumentDiff|null>(null); const [diffBusy,setDiffBusy]=useState(false);
  const [world,setWorld]=useState(defaultWorld); const [chapter,setChapter]=useState(defaultChapter); const [run,setRun]=useState('');
  const [progress,setProgress]=useState(0); const [message,setMessage]=useState('准备就绪'); const [issues,setIssues]=useState<Issue[]>([]);
  const [records,setRecords]=useState<RecordRow[]>([]); const [warnings,setWarnings]=useState<string[]>([]); const [filter,setFilter]=useState('all');
  const [busy,setBusy]=useState(false); const [runInfo,setRunInfo]=useState<RunInfo|null>(null); const [action,setAction]=useState('');
  const [feedbacks,setFeedbacks]=useState<Record<string,FeedbackState|null>>({}); const [notes,setNotes]=useState<Record<string,string>>({});
  const [diagnostics,setDiagnostics]=useState<Diagnostics>({});
  const [graph,setGraph]=useState<GraphResponse|null>(null); const [timeline,setTimeline]=useState<TimelineResponse|null>(null);
  const [visualTab,setVisualTab]=useState<'graph'|'timeline'>('graph'); const [focusedIssue,setFocusedIssue]=useState<string|null>(null);
  const streamRef=useRef<EventSource|null>(null);
  const visibleIssues=useMemo(()=>filter==='all'?issues:issues.filter(x=>x.category===filter),[issues,filter]);
  const selectedProject=projects.find(row=>row.id===project);

  async function readJson(response:Response){if(!response.ok){const body=await response.json().catch(()=>({detail:response.statusText}));throw new Error(body.detail||response.statusText)}return response.json()}
  async function loadProjects(){const rows=await readJson(await fetch(`${API}/api/v1/projects`));setProjects(rows)}
  async function loadProject(id:string){
    setProject(id);setDocumentDiff(null); if(!id){setDocs([]);setRuns([]);setDiffFrom('');setDiffTo('');return}
    const [documents,history]=await Promise.all([
      readJson(await fetch(`${API}/api/v1/projects/${id}/documents?include_history=true`)),
      readJson(await fetch(`${API}/api/v1/projects/${id}/analysis-runs`)),
    ]); const documentRows=documents as Doc[];setDocs(documentRows); setRuns(history);
    const versionGroup=(documentRows.map(row=>documentRows.filter(candidate=>candidate.name.toLocaleLowerCase()===row.name.toLocaleLowerCase())).find(group=>group.length>=2)||[]).sort((a,b)=>a.version-b.version);
    if(versionGroup.length>=2){setDiffFrom(versionGroup[0].id);setDiffTo(versionGroup[versionGroup.length-1].id)}else{setDiffFrom('');setDiffTo('')}
    if(history[0]) await restoreRun(history[0]);
  }
  async function loadFeedback(rows:Issue[]){
    const pairs=await Promise.all(rows.map(async issue=>{const value=await readJson(await fetch(`${API}/api/v1/issues/${issue.id}/feedback`));return [issue.id,value.latest] as const}));
    setFeedbacks(Object.fromEntries(pairs));
  }
  async function loadCompleted(runId:string){
    const [ir,rr,sr,dr,gr,tr]=await Promise.all([fetch(`${API}/api/v1/analysis-runs/${runId}/issues`),fetch(`${API}/api/v1/analysis-runs/${runId}/records`),fetch(`${API}/api/v1/analysis-runs/${runId}`),fetch(`${API}/api/v1/analysis-runs/${runId}/diagnostics`),fetch(`${API}/api/v1/analysis-runs/${runId}/graph`),fetch(`${API}/api/v1/analysis-runs/${runId}/timeline`)]);
    const loadedIssues=await readJson(ir); const rec=await readJson(rr); const status=await readJson(sr); const diag=await readJson(dr); const graphData=await readJson(gr); const timelineData=await readJson(tr);
    setIssues(loadedIssues);setRecords(rec.records);setWarnings(rec.warnings);setRunInfo(status);setDiagnostics(diag);setGraph(graphData);setTimeline(timelineData);await loadFeedback(loadedIssues);
  }
  function subscribe(runId:string,projectId:string){
    streamRef.current?.close(); const es=new EventSource(`${API}/api/v1/analysis-runs/${runId}/events`); streamRef.current=es;
    es.addEventListener('progress',event=>{const data=JSON.parse((event as MessageEvent).data);setProgress(data.progress);setMessage(data.message)});
    es.addEventListener('terminal',async event=>{const data=JSON.parse((event as MessageEvent).data);es.close();streamRef.current=null;setBusy(false);setAction('');setMessage(data.error?`${data.status}：${data.error}`:`任务状态：${data.status}`);const status=await readJson(await fetch(`${API}/api/v1/analysis-runs/${runId}`));setRunInfo(status);if(data.status==='completed')await loadCompleted(runId);await Promise.all([loadProjects(),loadProjectRuns(projectId)])});
    es.onerror=()=>{es.close();streamRef.current=null;setMessage('SSE 连接中断，可从运行历史恢复');setBusy(false)};
  }
  async function loadProjectRuns(id:string){if(!id)return;setRuns(await readJson(await fetch(`${API}/api/v1/projects/${id}/analysis-runs`)))}
  async function restoreRun(info:RunInfo,subscribeIfActive=true){
    setRun(info.id);setRunInfo(info);setProgress(info.status==='completed'?100:0);setMessage(info.error||`历史任务：${info.status}`);
    setIssues([]);setRecords([]);setWarnings([]);setFeedbacks({});setDiagnostics({});setGraph(null);setTimeline(null);setFocusedIssue(null);
    if(info.status==='completed')await loadCompleted(info.id);
    else if(subscribeIfActive&&['queued','running'].includes(info.status)){setBusy(true);subscribe(info.id,info.project_id)}
  }
  async function runProject(id:string){
    if(!id)throw new Error('请先选择项目');setProject(id);setBusy(true);setIssues([]);setRecords([]);setDiagnostics({});setGraph(null);setTimeline(null);setFocusedIssue(null);setRunInfo(null);setProgress(0);setMessage('任务已提交（若服务器启用模型，可能消耗 Token）');
    const created=await readJson(await fetch(`${API}/api/v1/projects/${id}/analysis-runs`,{method:'POST'}));setRun(created.id);subscribe(created.id,id);await loadProjectRuns(id);
  }
  async function createProject(){try{if(!projectName.trim())return;const created=await readJson(await fetch(`${API}/api/v1/projects`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:projectName.trim(),description:'本地工作台项目'})}));setProjectName('');await loadProjects();await loadProject(created.id)}catch(error){setMessage(String(error))}}
  async function uploadDocuments(){try{
    if(!project||files.length===0)return;if(replaceId&&files.length!==1)throw new Error('替换版本时只能选择一个同名文件');
    setAction('upload');for(const file of files){const form=new FormData();form.append('file',file);if(replaceId)form.append('replace_document_id',replaceId);await readJson(await fetch(`${API}/api/v1/projects/${project}/documents`,{method:'POST',body:form}))}
    setFiles([]);setReplaceId('');setMessage('文档已上传；同名文件已自动生成新版本');await loadProject(project);await loadProjects();
  }catch(error){setMessage(String(error))}finally{setAction('')}}
  async function demo(kind:'simple'|'advanced'){try{setAction(kind);const url=kind==='advanced'?'/api/v1/demo/advanced':'/api/v1/demo';const created=await readJson(await fetch(`${API}${url}`,{method:'POST'}));await loadProjects();await loadProject(created.id);await runProject(created.id)}catch(error){setBusy(false);setAction('');setMessage(String(error))}}
  async function custom(){try{setBusy(true);setAction('custom');const created=await readJson(await fetch(`${API}/api/v1/projects`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:`自然文本审查 ${new Date().toLocaleString()}`,description:'网页粘贴文本'})}));for(const [name,content] of [['world.md',world],['chapter.md',chapter]])await readJson(await fetch(`${API}/api/v1/projects/${created.id}/documents/text`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,content})}));await loadProjects();await loadProject(created.id);await runProject(created.id)}catch(error){setBusy(false);setAction('');setMessage(String(error))}}
  async function cancel(){try{await readJson(await fetch(`${API}/api/v1/analysis-runs/${run}/cancel`,{method:'POST'}));setMessage('取消请求已提交')}catch(error){setMessage(String(error))}}
  async function retry(){try{if(!run||!runInfo)return;const created=await readJson(await fetch(`${API}/api/v1/analysis-runs/${run}/retry`,{method:'POST'}));setRun(created.id);setBusy(true);setMessage('重试已提交（若服务器启用模型，可能消耗 Token）');subscribe(created.id,runInfo.project_id);await loadProjectRuns(runInfo.project_id)}catch(error){setMessage(String(error))}}
  async function submitFeedback(id:string,label:string){try{const comment=notes[id]||'';const value=await readJson(await fetch(`${API}/api/v1/issues/${id}/feedback`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({label,comment})}));setFeedbacks(current=>({...current,[id]:value}));setMessage(value.duplicate_ignored?'相同反馈已存在，未重复写入':'反馈已记录并保留审计历史')}catch(error){setMessage(String(error))}}
  function selectDiffFrom(id:string){
    setDiffFrom(id);setDocumentDiff(null);const source=docs.find(row=>row.id===id);const candidates=source?docs.filter(row=>row.name.toLocaleLowerCase()===source.name.toLocaleLowerCase()&&row.id!==id):[];setDiffTo(candidates.sort((a,b)=>b.version-a.version)[0]?.id||'');
  }
  async function compareVersions(){try{if(!project||!diffFrom||!diffTo)return;setDiffBusy(true);setDocumentDiff(await readJson(await fetch(`${API}/api/v1/projects/${project}/documents/diff?from_document_id=${encodeURIComponent(diffFrom)}&to_document_id=${encodeURIComponent(diffTo)}`)))}catch(error){setMessage(String(error));setDocumentDiff(null)}finally{setDiffBusy(false)}}

  useEffect(()=>{loadProjects().catch(error=>setMessage(String(error)));return()=>streamRef.current?.close()},[]);

  return <><header><span>LOREGUARD / LOCAL WORKSPACE</span><div><b>PROJECT WORKFLOW</b><i>证据先于判断 · 失败安全降级</i></div></header><main>
    <section className="hero"><div><p className="eyebrow">NARRATIVE CONSISTENCY REVIEW</p><h1>从项目版本<br/>追踪每次审查。</h1><p>创建或选择项目，管理文档版本，恢复历史运行，并让每条人工反馈留下审计记录。</p></div><aside><h2>原创样例</h2><p className="tokenWarn">⚠ 启动任何分析时，如果服务端启用了模型抽取，均可能消耗 Token；未启用时自动使用本地基线。</p><button className="primary" disabled={busy} onClick={()=>demo('simple')}>{action==='simple'?'分析中…':'运行简单样例（可能消耗 Token）'}</button><button className="wide" disabled={busy} onClick={()=>demo('advanced')}>{action==='advanced'?'分析中…':'运行复杂样例（可能消耗 Token）'}</button></aside></section>

    <section className="projectPanel"><div className="sectionHead"><div><p className="eyebrow">PROJECTS</p><h2>本地项目工作台</h2></div><button onClick={()=>loadProjects()}>刷新列表</button></div><div className="projectControls"><select value={project} onChange={event=>loadProject(event.target.value)}><option value="">选择已有项目</option>{projects.map(row=><option key={row.id} value={row.id}>{row.name} · {row.active_document_count} 文档 · {row.latest_run?.status||'未运行'}</option>)}</select><input value={projectName} onChange={event=>setProjectName(event.target.value)} placeholder="新项目名称"/><button onClick={createProject}>新建项目</button><button className="primary" disabled={!project||busy||docs.filter(row=>row.active).length===0} onClick={()=>runProject(project)}>分析当前项目（可能消耗 Token）</button></div>{selectedProject&&<p className="hint">当前：{selectedProject.name} · {selectedProject.description||'无描述'}</p>}
      <div className="uploadRow"><input type="file" multiple accept=".md,.txt,.json" onChange={event=>setFiles(Array.from(event.target.files||[]))}/><select value={replaceId} onChange={event=>setReplaceId(event.target.value)}><option value="">同名自动新版本 / 新文件</option>{docs.filter(row=>row.active).map(row=><option key={row.id} value={row.id}>明确替换 {row.name} v{row.version}</option>)}</select><button disabled={!project||files.length===0||action==='upload'} onClick={uploadDocuments}>上传 {files.length||''} 个文件</button></div>
      <div className="tables"><div><h3>文档版本</h3><table><thead><tr><th>文件</th><th>版本</th><th>状态</th><th>创建时间</th></tr></thead><tbody>{docs.map(row=><tr key={row.id}><td>{row.name}</td><td>v{row.version}</td><td><span className={`badge ${row.active?'ok':'muted'}`}>{row.active?'active':'history'}</span></td><td>{new Date(row.created_at).toLocaleString()}</td></tr>)}</tbody></table></div><div><h3>运行历史</h3><table><thead><tr><th>时间</th><th>状态</th><th>Token</th><th></th></tr></thead><tbody>{runs.map(row=><tr key={row.id}><td>{new Date(row.created_at).toLocaleString()}</td><td><span className={`badge ${row.status}`}>{row.status}</span></td><td>{row.prompt_tokens+row.completion_tokens}</td><td><button onClick={()=>restoreRun(row)}>恢复</button></td></tr>)}</tbody></table></div></div>
      <div className="diffPanel"><div className="sectionHead"><div><p className="eyebrow">VERSION DIFF</p><h3>版本内容差异</h3></div><small>本地行级比较，不调用模型、不消耗 Token</small></div><div className="diffControls"><select value={diffFrom} onChange={event=>selectDiffFrom(event.target.value)}><option value="">选择旧版本</option>{docs.map(row=><option key={row.id} value={row.id}>{row.name} · v{row.version}</option>)}</select><span>→</span><select value={diffTo} onChange={event=>{setDiffTo(event.target.value);setDocumentDiff(null)}}><option value="">选择新版本</option>{docs.filter(row=>{const source=docs.find(item=>item.id===diffFrom);return source&&row.id!==source.id&&row.name.toLocaleLowerCase()===source.name.toLocaleLowerCase()}).map(row=><option key={row.id} value={row.id}>{row.name} · v{row.version}</option>)}</select><button disabled={!diffFrom||!diffTo||diffBusy} onClick={compareVersions}>{diffBusy?'比较中…':'查看差异'}</button></div>{documentDiff&&<div className="diffResult"><div className="diffSummary"><b>v{documentDiff.from_document.version} → v{documentDiff.to_document.version}</b><span className="diffAdded">+{documentDiff.summary.added_lines}</span><span className="diffRemoved">−{documentDiff.summary.removed_lines}</span><span>{documentDiff.summary.changed_hunks} 个变更区块</span><small>比较 {documentDiff.summary.compared_old_lines}/{documentDiff.summary.old_total_lines} → {documentDiff.summary.compared_new_lines}/{documentDiff.summary.new_total_lines} 行</small></div>{documentDiff.warnings.map((warning,index)=><p className="visualWarning" key={index}>{warning}</p>)}{documentDiff.hunks.length===0?<div className="visualEmpty">两个版本的文本内容相同。</div>:<div className="diffHunks">{documentDiff.hunks.map((hunk,index)=><div className="diffHunk" key={index}><div className="diffHeader">@@ -{hunk.old_start},{hunk.old_lines} +{hunk.new_start},{hunk.new_lines} @@</div>{hunk.lines.map((line,lineIndex)=><div className={`diffLine ${line.type}`} key={lineIndex}><code>{line.old_line??''}</code><code>{line.new_line??''}</code><b>{line.type==='added'?'+':line.type==='removed'?'−':' '}</b><pre>{line.content||' '}</pre></div>)}</div>)}</div>}</div>}</div>
    </section>

    <section className="workspace"><div className="sectionHead"><div><p className="eyebrow">QUICK TEXT</p><h2>快速自然文本实验台</h2></div><button disabled={busy} onClick={custom}>{action==='custom'?'分析中…':'创建项目并运行（可能消耗 Token）'}</button></div><div className="editors"><label><span>权威设定 / world.md</span><textarea value={world} onChange={event=>setWorld(event.target.value)}/></label><label><span>待审章节 / chapter.md</span><textarea value={chapter} onChange={event=>setChapter(event.target.value)}/></label></div></section>

    <section className="status"><div><small>PROJECT</small><code>{project||'not-selected'}</code></div><div><small>RUN</small><code>{run||'not-started'}</code></div><div className="meter"><i style={{width:`${progress}%`}}/></div><strong>{progress}% · {message}</strong><div className="runActions">{busy&&<button onClick={cancel}>取消</button>}{runInfo&&['failed','cancelled'].includes(runInfo.status)&&<button onClick={retry}>重试（可能消耗 Token）</button>}</div></section>
    <section className="summary"><div><small>当前文档</small><b>{docs.filter(row=>row.active).length}</b></div><div><small>抽取记录</small><b>{records.length}</b></div><div><small>一致性问题</small><b>{issues.length}</b></div><div><small>模型 Token</small><b>{runInfo?runInfo.prompt_tokens+runInfo.completion_tokens:0}</b><small>成本：{runInfo?.estimated_cost_usd==null?'未配置':`$${runInfo.estimated_cost_usd.toFixed(6)}`}</small></div></section>
    <section className="records"><details><summary>查看系统实际抽取记录（{records.length} 条）</summary><div className="recordGrid">{records.map(row=><div key={row.id}><span>{row.kind}</span><code>{Object.entries(row.attrs).map(([key,value])=>`${key}=${value}`).join(' · ')}</code><small>{row.evidence.document_name}:{row.evidence.line_start}</small></div>)}</div>{warnings.length>0&&<div className="warnings"><b>抽取提示</b>{warnings.map((warning,index)=><p key={index}>{warning}</p>)}</div>}</details></section>
    <section className="records"><details><summary>分析诊断摘要</summary><div className="recordGrid"><div><span>CHUNKING</span><code>{diagnostics.chunking?.total_chunks||0} 个分块</code><small>{diagnostics.chunking?.documents?.map(row=>`${row.document_name}: ${row.chunk_count}`).join(' · ')||'暂无诊断'}</small></div><div><span>ALIASES</span><code>{diagnostics.aliases?.declaration_count||0} 条声明 · {diagnostics.aliases?.trace_count||0} 次归一化</code><small>只接受显式又名/简称/化名/代号声明</small></div><div><span>RETRIEVAL</span><code>{diagnostics.retrieval?.candidate_count||0} 个候选 · {diagnostics.retrieval?.consumed_count||0} 个被检查器消费</code><small>{diagnostics.retrieval?.boundary||'本地稳定哈希与实体图候选'}</small></div><div><span>TIMINGS</span><code>总计 {diagnostics.timings?.total_ms?.toFixed(1)||'0.0'} ms · 首事件 {diagnostics.timings?.first_progress_ms?.toFixed(1)||'0.0'} ms</code><small>chunk {diagnostics.timings?.chunk_ms||0} · extract {diagnostics.timings?.extract_ms||0} · index {diagnostics.timings?.index_ms||0} · check {diagnostics.timings?.check_ms||0} · report {diagnostics.timings?.report_ms||0}</small></div></div></details></section>
    <section className="visualization"><div className="sectionHead"><div><p className="eyebrow">NARRATIVE VIEW</p><h2>关系图与时间线</h2></div><div className="visualTabs"><button className={visualTab==='graph'?'active':''} onClick={()=>setVisualTab('graph')}>关系图</button><button className={visualTab==='timeline'?'active':''} onClick={()=>setVisualTab('timeline')}>时间线</button></div></div>{visualTab==='graph'?<Suspense fallback={<div className="visualEmpty">正在加载关系图…</div>}><RelationGraph data={graph} focusedIssueId={focusedIssue}/></Suspense>:<NarrativeTimeline data={timeline} focusedIssueId={focusedIssue}/>}</section>
    <section className="issues"><div className="sectionHead"><div><p className="eyebrow">EVIDENCE REPORT</p><h2>问题报告 <em>{visibleIssues.length}</em></h2></div><select value={filter} onChange={event=>setFilter(event.target.value)}><option value="all">全部类别</option>{Object.entries(categoryNames).map(([key,value])=><option key={key} value={key}>{value}</option>)}</select></div>{issues.length===0&&!busy&&<div className="empty">选择项目运行，或从历史任务恢复报告。</div>}{visibleIssues.map((issue,index)=><article key={issue.id} className={focusedIssue===issue.id?'focusedIssue':''} onClick={()=>setFocusedIssue(issue.id)}><div className="rank">{String(index+1).padStart(2,'0')}</div><div><p className="tag">{categoryNames[issue.category]||issue.category} · {issue.severity} · {(issue.confidence*100).toFixed(0)}%</p><h3>{issue.title}</h3><p>{issue.explanation}</p><div className="evidence">{issue.evidence.map((evidence,index)=><blockquote key={index}><b>{evidence.document_name}:{evidence.line_start}</b>{evidence.text}</blockquote>)}</div><p className="suggestion">建议：{issue.suggestion}</p><div className="feedbackState">当前反馈：{feedbacks[issue.id]?feedbackNames[feedbacks[issue.id]!.label]||feedbacks[issue.id]!.label:'未反馈'}</div><input className="note" value={notes[issue.id]||''} onChange={event=>setNotes(current=>({...current,[issue.id]:event.target.value}))} placeholder="可选备注（会进入审计历史）"/><div className="feedback">{Object.entries(feedbackNames).map(([label,title])=><button key={label} disabled={feedbacks[issue.id]?.label===label&&(feedbacks[issue.id]?.comment||'')===(notes[issue.id]||'')} onClick={()=>submitFeedback(issue.id,label)}>{title}</button>)}</div></div></article>)}</section>
  </main><footer>原创演示文本 · 本地项目/版本/运行历史 · Provider 异常时安全降级</footer></>;
}
