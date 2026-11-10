import { test, expect, type Page } from "@playwright/test";
import { fulfillLegacyMaterialImpact, fulfillLegacyRunCatalog } from "./materialImpactFixture";

const project = "journey-project", runId = "journey-run", otherRun = "journey-other", now = "2026-11-10T00:00:00Z";
const item = { key: "ij_key", name: "铜钥匙", story_scope: "main", record_count: 1, issue_ids: ["item-issue"] };
const evidence = { document_id: "doc-v1", document_name: "交接.md", line_start: 1, line_end: 1, text: "09:30，许岩把铜钥匙交给陈蔚。" };
const event = { record_id: "record-1", kind: "transfer", holder: "陈蔚", transfer_from: "许岩", time: "09:30", through_time: null, retention_scope: null,
  basis: "explicit", source_status: "verified", semantic_status: "supported", evidence, document_version: 1 };
const run = (id = runId) => ({ id, project_id: project, status: "completed", progress: 100, created_at: now, completed_at: now,
  prompt_tokens: 0, completion_tokens: 0, estimated_cost_usd: 0, input_snapshot_available: true,
  input_documents: [{ document_id: "doc-v1", document_name: "交接.md", document_version: 1, content_sha256: "a".repeat(64), ordinal: 0, document_role: "chapter", story_scope: "main", batch_role: "target" }],
  review_batch: { mode: "draft_review", sensitivity: "balanced", character_review_enabled: false } });
const issue = { id: "item-issue", category: "item_ownership", severity: "high", confidence: .9, title: "物品引用跳转测试", explanation: "仅用于界面跳转。", suggestion: "阅读原文", evidence: [evidence], metadata: { item: "铜钥匙" } };
const detail = (id = runId) => ({ schema_version: "item-journey-v1", run_id: id, project_id: project, item,
  summary: { status: "ordered_evidence", text: "本次最后有据记录：09:30，许岩将铜钥匙交给陈蔚。", latest_record_ids: [event.record_id] },
  events: { items: [event], offset: 0, limit: 50, total: 1, has_more: false }, relations: [], relation_evidence: [],
  coverage: { processed_records: 1, total_records: 1, truncated: false, source_unavailable: 0, proof_unavailable: 0, unresolved_time_pairs: 0, run_partial: false, relation_proof_unavailable: 0 } });
async function fixture(page: Page, withIssue = false) {
 const state = { requests: [] as string[], writes: [] as string[], unexpected: [] as string[], failures: 0, delayed: false, release: undefined as (()=>void)|undefined };
 await page.route("**/api/v1/**", async route => {
  const url = new URL(route.request().url()), p = url.pathname, method = route.request().method();state.requests.push(`${method} ${p}${url.search}`);
  if (method !== "GET") state.writes.push(`${method} ${p}`);
  if (await fulfillLegacyMaterialImpact(route, [runId,otherRun].map(runId=>({projectId:project,runId})))) return;
  if (await fulfillLegacyRunCatalog(route,[{projectId:project,runs:[run(),run(otherRun)]}])) return;
  let body: unknown;
  if(p==="/api/v1/auth/me") body={mode:"anonymous",user:{id:"qa",email:"",display_name:"作者"},workspace:{id:"space",name:"原创空间",kind:"personal",role:"owner"}};
  else if(p==="/api/v1/account/model-provider") body={configured:false,revision:0};
  else if(p==="/api/v1/project-catalog") body={page:1,page_size:40,total:1,items:[{id:project,name:"交接故事",description:"",created_at:now,active_document_count:0,latest_run:run()}]};
  else if(p===`/api/v1/projects/${project}/documents`) body=[];
  else if(p===`/api/v1/projects/${project}/analysis-runs`) body=[run(),run(otherRun)];
  else if(p==="/api/v1/issues/item-issue/feedback") body={latest:null};
  else {
   const id=[runId,otherRun].find(id=>p===`/api/v1/analysis-runs/${id}`||p.startsWith(`/api/v1/analysis-runs/${id}/`));
   const suffix=id?p.slice(`/api/v1/analysis-runs/${id}`.length):"invalid";
   if(!id) { state.unexpected.push(p);await route.fulfill({status:404,json:{detail:"unexpected endpoint"}});return; }
   if(!suffix) body=run(id);
   else if(suffix==="/issues") body=withIssue?[issue]:[];
   else if(suffix==="/records") body={records:[],warnings:[]};
   else if(suffix==="/diagnostics") body={};
   else if(suffix==="/clarifications") body=[];
   else if(suffix==="/review-clues") body={items:[],truncated:false,unavailable_count:0};
   else if(suffix==="/provisional-clues") body={items:[],truncated:false};
   else if(suffix==="/item-journeys") {
    expect(url.searchParams.get("limit")).toBe("30");expect(url.searchParams.get("offset")).toBe("0");
    if(state.failures){state.failures--;await route.fulfill({status:503,json:{detail:"temporarily unavailable"}});return;}
    if(state.delayed&&id===runId)await new Promise<void>(resolve=>{state.release=resolve;});
    const matches=id===runId&&(!url.searchParams.get("q")||"铜钥匙".includes(url.searchParams.get("q")!));
    body={schema_version:"item-journey-v1",run_id:id,project_id:project,items:matches?[item]:[],page:{offset:0,limit:30,total:matches?1:0,has_more:false},coverage:{scanned_records:matches?1:0,total_records:matches?1:0,truncated:false}};
   } else if(suffix==="/item-journeys/ij_key"&&id===runId) { expect(url.searchParams.get("limit")).toBe("50");body=detail(id); }
   else if(suffix==="/evidence-preview"&&id===runId) {
    const kind=url.searchParams.get("kind"),key=url.searchParams.get("item_id");
    expect(kind==='record'&&key==='record-1'||withIssue&&kind==='issue'&&key==='item-issue').toBe(true);
    body={run:{id,project_id:project},reference:{...evidence,kind,item_id:key,evidence_index:0},source:{availability:"full_context",provenance:"run_input",source_run_id:id,document_version:1,content_sha256:"a".repeat(64),char_count:evidence.text.length,line_count:1},lines:[{line_number:1,text:evidence.text,is_evidence:true}],page:{offset:0,limit:120,total:1,has_more:false},message:null};
   } else {state.unexpected.push(p);await route.fulfill({status:404,json:{detail:"unexpected endpoint"}});return;}
  }
  await route.fulfill({json:body});
 });return state;
}
const path=(id=runId)=>`/app/projects/${project}/runs/${id}/visuals?view=items`;

test("零正式问题也能查物品并阅读冻结原文，窄屏可返回历程",async({page})=>{
 const state=await fixture(page);await page.goto(path());
 await page.getByRole("button",{name:/^铜钥匙/}).click();
 const section=page.getByRole("region",{name:"所选物品历程"});await expect(section).toContainText("09:30，许岩将铜钥匙交给陈蔚");
 expect(state.requests.some(r=>r.includes("evidence-preview"))).toBe(false);
 await page.setViewportSize({width:375,height:900});await section.getByRole("button",{name:"阅读这条记录的原文",exact:true}).click();
 const reader=page.getByRole("region",{name:"物品记录冻结原文"});await expect(reader).toContainText("冻结 v1");await expect(reader).toContainText(evidence.text);
 expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
 await page.getByRole("button",{name:"返回物品历程",exact:true}).click();await expect(section.getByRole("heading",{name:"有出处的历程记录"})).toBeFocused();
 const top=await section.getByRole("heading",{name:"有出处的历程记录"}).boundingBox();
 const sticky=await page.locator(".topBar").boundingBox();expect(top!.y+top!.height).toBeGreaterThan(sticky!.y+sticky!.height);
 expect(state.writes).toEqual([]);expect(state.unexpected).toEqual([]);
});

test("物品问题直达实际关联项，搜索空结果不冒充无交接",async({page})=>{
 const state=await fixture(page,true);await page.goto(`/app/projects/${project}/runs/${runId}/report`);
 await page.getByRole("button",{name:"查看相关物品去向",exact:true}).click();
 await expect(page).toHaveURL(/view=items&item=ij_key&item_issue=item-issue/);await expect(page.getByRole("region",{name:"所选物品历程"})).toContainText("许岩将铜钥匙交给陈蔚");
 await page.getByRole("button",{name:"查看全部物品",exact:true}).click();await page.getByRole("searchbox",{name:"查找物品"}).fill("不存在的名称");await page.getByRole("button",{name:"查找",exact:true}).click();
 await expect(page.getByRole("region",{name:"物品目录"})).toContainText("未找到匹配物品，可换个原文名称查找");expect(state.writes).toEqual([]);expect(state.unexpected).toEqual([]);
});

test("目录失败可重试，切换报告后迟到结果不能混入当前范围",async({page})=>{
 const state=await fixture(page);state.failures=1;await page.goto(path());
 await expect(page.getByRole("alert")).toContainText("页面没有改用其他报告或当前文稿");await page.getByRole("button",{name:"重试读取物品目录",exact:true}).click();await expect(page.getByRole("button",{name:/^铜钥匙/})).toBeVisible();
 state.delayed=true;await page.getByRole("searchbox",{name:"查找物品"}).fill("铜");await page.getByRole("button",{name:"查找",exact:true}).click();await expect.poll(()=>!!state.release).toBe(true);
 await page.goto(path(otherRun));await expect(page.getByRole("region",{name:"物品目录"})).toContainText("这份报告没有可整理的物品记录");state.release?.();
 await expect(page.getByRole("button",{name:/^铜钥匙/})).toHaveCount(0);expect(state.writes).toEqual([]);expect(state.unexpected).toEqual([]);
});
