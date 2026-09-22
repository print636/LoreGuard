import {
  type FormEvent,
  type MouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ApiError } from "../../api/client";
import type { WorkspaceView } from "../../routing";
import {
  fetchCharacter,
  fetchCharacters,
  fetchDriftIssues,
  fetchProfileCandidate,
  fetchProfileCandidates,
  submitCandidateDecision,
} from "./api";
import CandidateReview from "./CandidateReview";
import CharacterDriftList from "./CharacterDriftList";
import CharacterProfile from "./CharacterProfile";
import CharacterRoster from "./CharacterRoster";
import {
  describeCoverage,
  describeReadiness,
} from "./presentation";
import {
  characterRouteStateFromSearch,
  characterSearch,
  type CharacterRouteState,
} from "./routeState";
import type {
  CandidateDecision,
  CandidatePage,
  CharacterDetail,
  CharacterPage,
  CharacterReadiness,
  DriftIssuePage,
  ProfileCandidate,
} from "./types";

type CharacterWorkspaceProps = {
  projectId: string;
  documentCount: number;
  completedRunCount: number;
  projectLoading: boolean;
  baselineContextReady: boolean;
  baselineContextDetail: string;
  routeSearch: string;
  onRouteChange: (search: string, replace?: boolean) => void;
  onNavigateWorkspace: (view: WorkspaceView) => void;
  onOpenProjectCenter: () => void;
};

function requestError(error: unknown): string {
  if (error instanceof DOMException && error.name === "AbortError") return "";
  if (error instanceof ApiError) {
    if (error.status === 409) {
      return "档案已经在其他页面更新。页面不会覆盖新状态，请刷新后重新核对。";
    }
    if (error.status === 429) return "请求过于频繁，请稍后重试。";
    if (error.status === 404) return "这项角色资料不存在，或不属于当前项目。";
    return error.message || "请求没有完成，请重试。";
  }
  if (error instanceof TypeError) {
    return "服务返回的角色资料格式不兼容。页面没有把它当作空结果，请重试或查看运行审计。";
  }
  return "服务暂时不可用，请稍后重试。";
}

function normalLink(event: MouseEvent<HTMLAnchorElement>): boolean {
  return !(
    event.defaultPrevented ||
    event.button !== 0 ||
    event.metaKey ||
    event.ctrlKey ||
    event.shiftKey ||
    event.altKey
  );
}

export default function CharacterWorkspace({
  projectId,
  documentCount,
  completedRunCount,
  projectLoading,
  baselineContextReady,
  baselineContextDetail,
  routeSearch,
  onRouteChange,
  onNavigateWorkspace,
  onOpenProjectCenter,
}: CharacterWorkspaceProps) {
  const route = useMemo(
    () => characterRouteStateFromSearch(routeSearch),
    [routeSearch],
  );
  const [searchDraft, setSearchDraft] = useState(route.query);
  const [characterPage, setCharacterPage] = useState<CharacterPage | null>(null);
  const [characterDetail, setCharacterDetail] = useState<CharacterDetail | null>(null);
  const [candidatePage, setCandidatePage] = useState<CandidatePage | null>(null);
  const [candidate, setCandidate] = useState<ProfileCandidate | null>(null);
  const [driftPage, setDriftPage] = useState<DriftIssuePage | null>(null);
  const [listLoading, setListLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [sectionLoading, setSectionLoading] = useState(false);
  const [candidateLoading, setCandidateLoading] = useState(false);
  const [decisionBusy, setDecisionBusy] = useState<CandidateDecision | null>(null);
  const [listError, setListError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [sectionError, setSectionError] = useState("");
  const [candidateError, setCandidateError] = useState("");
  const [actionError, setActionError] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const [listReload, setListReload] = useState(0);
  const [detailReload, setDetailReload] = useState(0);
  const [sectionReload, setSectionReload] = useState(0);
  const [candidateReload, setCandidateReload] = useState(0);
  const pageTitleRef = useRef<HTMLHeadingElement | null>(null);
  const detailTitleRef = useRef<HTMLHeadingElement | null>(null);
  const detailRegionRef = useRef<HTMLElement | null>(null);

  const prerequisiteReadiness: CharacterReadiness | null = projectId
    ? documentCount === 0
      ? "no_documents"
      : completedRunCount === 0
        ? "no_completed_run"
        : null
    : null;
  const readiness = prerequisiteReadiness || characterPage?.readiness || "ready";
  const readinessView = describeReadiness(readiness);

  function pathFor(next: CharacterRouteState): string {
    return `/app/projects/${encodeURIComponent(projectId)}/characters${characterSearch(next)}`;
  }

  function navigate(
    next: Partial<CharacterRouteState>,
    replace = false,
  ) {
    onRouteChange(characterSearch({ ...route, ...next }), replace);
  }

  useEffect(() => {
    setSearchDraft(route.query);
  }, [route.query]);

  useEffect(() => {
    requestAnimationFrame(() => pageTitleRef.current?.focus());
  }, [projectId]);

  useEffect(() => {
    if (!route.characterId) return;
    requestAnimationFrame(() => detailRegionRef.current?.focus());
  }, [route.characterId]);

  useEffect(() => {
    if (
      !route.characterId ||
      !characterDetail ||
      characterDetail.character.id !== route.characterId
    ) return;
    requestAnimationFrame(() => detailTitleRef.current?.focus());
  }, [route.characterId, route.section, characterDetail?.character.id]);

  useEffect(() => {
    if (
      !projectId ||
      projectLoading ||
      documentCount === 0 ||
      completedRunCount === 0
    ) {
      setCharacterPage(null);
      setListError("");
      setListLoading(false);
      return;
    }
    const controller = new AbortController();
    setListLoading(true);
    setListError("");
    void fetchCharacters(
      projectId,
      { page: route.page, query: route.query },
      controller.signal,
    )
      .then(setCharacterPage)
      .catch((error) => {
        const message = requestError(error);
        if (message) setListError(message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setListLoading(false);
      });
    return () => controller.abort();
  }, [
    projectId,
    projectLoading,
    documentCount,
    completedRunCount,
    route.page,
    route.query,
    listReload,
  ]);

  useEffect(() => {
    setCharacterDetail(null);
    setDetailError("");
    if (!projectId || !route.characterId || readiness !== "ready") {
      setDetailLoading(false);
      return;
    }
    const controller = new AbortController();
    setDetailLoading(true);
    void fetchCharacter(projectId, route.characterId, controller.signal)
      .then(setCharacterDetail)
      .catch((error) => {
        const message = requestError(error);
        if (message) setDetailError(message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setDetailLoading(false);
      });
    return () => controller.abort();
  }, [projectId, route.characterId, readiness, detailReload]);

  useEffect(() => {
    setCandidatePage(null);
    setDriftPage(null);
    setSectionError("");
    if (!projectId || !route.characterId || readiness !== "ready") {
      setSectionLoading(false);
      return;
    }
    if (route.section === "profile") {
      setSectionLoading(false);
      return;
    }
    const controller = new AbortController();
    setSectionLoading(true);
    const request =
      route.section === "candidates"
        ? fetchProfileCandidates(
            projectId,
            route.characterId,
            { page: route.page },
            controller.signal,
          ).then(setCandidatePage)
        : fetchDriftIssues(
            projectId,
            route.characterId,
            { page: route.page },
            controller.signal,
          ).then(setDriftPage);
    void request
      .catch((error) => {
        const message = requestError(error);
        if (message) setSectionError(message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setSectionLoading(false);
      });
    return () => controller.abort();
  }, [
    projectId,
    route.characterId,
    route.section,
    route.page,
    readiness,
    sectionReload,
  ]);

  useEffect(() => {
    setCandidate(null);
    setCandidateError("");
    setActionError("");
    if (
      !projectId ||
      !route.characterId ||
      route.section !== "candidates" ||
      !route.candidateId ||
      readiness !== "ready"
    ) {
      setCandidateLoading(false);
      return;
    }
    const controller = new AbortController();
    setCandidateLoading(true);
    void fetchProfileCandidate(
      projectId,
      route.characterId,
      route.candidateId,
      controller.signal,
    )
      .then(setCandidate)
      .catch((error) => {
        const message = requestError(error);
        if (message) setCandidateError(message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setCandidateLoading(false);
      });
    return () => controller.abort();
  }, [
    projectId,
    route.characterId,
    route.section,
    route.candidateId,
    readiness,
    candidateReload,
  ]);

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    navigate({
      query: searchDraft.trim().slice(0, 120),
      page: 1,
      characterId: null,
      candidateId: null,
    });
  }

  async function decide(decision: CandidateDecision, comment: string) {
    if (!candidate || !route.characterId || decisionBusy) return;
    try {
      setDecisionBusy(decision);
      setActionError("");
      setAnnouncement("");
      const result = await submitCandidateDecision(
        projectId,
        route.characterId,
        candidate.id,
        {
          decision,
          comment: comment.trim(),
          expected_revision: candidate.revision,
        },
      );
      setCandidate(result.candidate);
      setAnnouncement(
        decision === "confirm"
          ? "归纳已确认并写入角色档案。"
          : "归纳已驳回，决定已保留在审核记录中。",
      );
      setListReload((value) => value + 1);
      setDetailReload((value) => value + 1);
      setSectionReload((value) => value + 1);
    } catch (error) {
      const message = requestError(error);
      setActionError(message);
      if (error instanceof ApiError && error.status === 409) {
        setCandidateReload((value) => value + 1);
        setDetailReload((value) => value + 1);
      }
    } finally {
      setDecisionBusy(null);
    }
  }

  const coverage = describeCoverage(
    characterDetail?.model_coverage || characterPage?.model_coverage || "unknown",
    characterDetail?.coverage_detail || characterPage?.coverage_detail,
  );

  if (!projectId) {
    return (
      <section className="characterWorkspace workspaceView">
        <div className="characterPrerequisite">
          <h1 ref={pageTitleRef} tabIndex={-1}>角色档案</h1>
          <p>先从项目中心选择一个项目，再查看由设定与历史剧情归纳的角色档案。</p>
          <button type="button" onClick={onOpenProjectCenter}>返回项目中心</button>
        </div>
      </section>
    );
  }

  return (
    <section className="characterWorkspace workspaceView" aria-labelledby="character-workspace-title">
      <header className="characterWorkspaceHead">
        <div>
          <p className="eyebrow">CHARACTER EVIDENCE ARCHIVE</p>
          <h1 id="character-workspace-title" ref={pageTitleRef} tabIndex={-1}>角色档案</h1>
          <p>AI 归纳候选只有经过确认，才会写入正式角色档案。</p>
        </div>
        {characterPage?.source_run_id && (
          <span className="characterRunRef">来源运行 {characterPage.source_run_id.slice(0, 8)}</span>
        )}
      </header>

      <div className="characterAnnouncement" aria-live="polite">
        {announcement}
      </div>
      {actionError && <div className="characterActionError" role="alert">{actionError}</div>}

      {projectLoading ? (
        <div className="characterWorkspaceLoading" aria-busy="true">正在读取项目资料…</div>
      ) : documentCount > 0 && !baselineContextReady ? (
        <div className="characterPrerequisite">
          <h2>先确认正式资料的上下文</h2>
          <p>{baselineContextDetail}</p>
          <button type="button" onClick={() => onNavigateWorkspace("projects")}>
            前往确认资料
          </button>
        </div>
      ) : readinessView ? (
        <div className="characterPrerequisite">
          <h2>{readinessView.title}</h2>
          <p>{readinessView.detail}</p>
          <button
            type="button"
            onClick={() => onNavigateWorkspace(readinessView.destination)}
          >
            {readinessView.action}
          </button>
        </div>
      ) : (
        <>
          {characterPage && (
            <section className={`characterCoverage overview ${coverage.tone}`} role="status">
              <b>{coverage.label}</b>
              <p>{coverage.detail}</p>
            </section>
          )}
          <div className="characterLayout">
            <CharacterRoster
              characters={characterPage?.items || []}
              selectedId={route.characterId}
              query={route.query}
              searchDraft={searchDraft}
              page={route.page}
              total={characterPage?.total || 0}
              hasMore={Boolean(characterPage?.has_more)}
              loading={listLoading}
              error={listError}
              onSearchDraftChange={setSearchDraft}
              onSearch={submitSearch}
              onClearSearch={() => navigate({ query: "", page: 1, characterId: null, candidateId: null })}
              hrefFor={(characterId) => pathFor({ ...route, characterId, section: "profile", candidateId: null, page: 1 })}
              onSelect={(characterId) => navigate({ characterId, section: "profile", candidateId: null, page: 1 })}
              onPage={(page) => navigate({ page, characterId: null, candidateId: null })}
              onRetry={() => setListReload((value) => value + 1)}
            />

            <section
              ref={detailRegionRef}
              className={`characterDetail ${route.characterId ? "selected" : ""}`}
              aria-label="角色档案详情"
              tabIndex={-1}
            >
              {!route.characterId ? (
                <div className="characterSelectPrompt">
                  <h2>选择角色查看档案</h2>
                  <p>档案、待确认归纳和角色漂移都保留原文证据与作用域。</p>
                </div>
              ) : detailError ? (
                <div className="characterPanelError" role="alert">
                  <b>角色档案没有加载</b>
                  <p>{detailError}</p>
                  <button type="button" onClick={() => setDetailReload((value) => value + 1)}>重试</button>
                </div>
              ) : detailLoading || !characterDetail ? (
                <div className="characterInlineLoading" aria-busy="true">正在读取角色档案…</div>
              ) : (
                <>
                  <button
                    className="characterMobileBack"
                    type="button"
                    onClick={() => navigate({ characterId: null, candidateId: null, page: 1 })}
                  >
                    返回角色列表
                  </button>
                  <header className="characterIdentity">
                    <span className="characterAvatar large" aria-hidden="true">
                      {characterDetail.character.canonical_name.slice(0, 1)}
                    </span>
                    <div>
                      <h2 ref={detailTitleRef} tabIndex={-1}>
                        {characterDetail.character.canonical_name}
                      </h2>
                      <p>
                        {characterDetail.character.aliases.length
                          ? `别名：${characterDetail.character.aliases.join("、")}`
                          : "尚未记录别名"}
                      </p>
                    </div>
                    <span>
                      {characterDetail.character.profile_revision === null
                        ? "档案版本未提供"
                        : `档案修订 ${characterDetail.character.profile_revision}`}
                    </span>
                  </header>

                  <nav className="characterSectionNav" aria-label="角色档案视图">
                    {([
                      ["profile", "已确认档案", characterDetail.character.confirmed_item_count],
                      ["candidates", "待确认归纳", characterDetail.character.pending_candidate_count],
                      ["drift", "漂移问题", characterDetail.character.drift_issue_count],
                    ] as const).map(([section, label, count]) => {
                      const next = { ...route, section, candidateId: null, page: 1 };
                      return (
                        <a
                          key={section}
                          href={pathFor(next)}
                          aria-current={route.section === section ? "page" : undefined}
                          onClick={(event) => {
                            if (!normalLink(event)) return;
                            event.preventDefault();
                            navigate({ section, candidateId: null, page: 1 });
                          }}
                        >
                          {label} <span>{count === null ? "—" : count}</span>
                        </a>
                      );
                    })}
                  </nav>

                  <div className="characterSectionBody">
                    {route.section === "profile" && (
                      <CharacterProfile items={characterDetail.profile_items} />
                    )}
                    {route.section === "candidates" && (
                      <CandidateReview
                        page={candidatePage}
                        selected={candidate}
                        selectedId={route.candidateId}
                        loading={sectionLoading}
                        detailLoading={candidateLoading}
                        error={sectionError}
                        detailError={candidateError}
                        decisionBusy={decisionBusy}
                        pageNumber={route.page}
                        hrefForCandidate={(candidateId) => pathFor({ ...route, candidateId })}
                        onSelect={(candidateId) => navigate({ candidateId })}
                        onPage={(page) => navigate({ page, candidateId: null })}
                        onRetry={() => setSectionReload((value) => value + 1)}
                        onRetryDetail={() => setCandidateReload((value) => value + 1)}
                        onBack={() => navigate({ candidateId: null })}
                        onDecision={(decision, comment) => void decide(decision, comment)}
                      />
                    )}
                    {route.section === "drift" && (
                      <CharacterDriftList
                        projectId={projectId}
                        page={driftPage}
                        pageNumber={route.page}
                        loading={sectionLoading}
                        error={sectionError}
                        onPage={(page) => navigate({ page })}
                        onRetry={() => setSectionReload((value) => value + 1)}
                      />
                    )}
                  </div>
                </>
              )}
            </section>
          </div>
        </>
      )}
    </section>
  );
}
