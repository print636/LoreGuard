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
  createCharacterTraitAxis,
  fetchCharacter,
  fetchCharacterTraitAxes,
  fetchCharacters,
  fetchDriftIssues,
  fetchProfileCandidate,
  fetchProfileCandidates,
  fetchWithdrawnProfileCandidates,
  fetchSourceNeighbors,
  setAxisPositiveProposition,
  submitCandidateDecision,
} from "./api";
import CandidateReview from "./CandidateReview";
import { appendAxisPage } from "./axisReview";
import {
  advanceReviewScope,
  isCurrentReviewRequest,
  isCurrentReviewScope,
  reviewScopeKey,
} from "./reviewScope";
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
  CharacterTraitAxis,
  DriftIssuePage,
  ProfileCandidate,
  CharacterProfileItem,
  SourceNeighborPage,
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
    const envelope = error.detail;
    const detail = envelope && typeof envelope === "object" && "detail" in envelope
      ? envelope.detail
      : null;
    const code = detail && typeof detail === "object" && "code" in detail
      ? detail.code
      : null;
    if (code === "character_trait_axis_version_conflict") return "作者轴版本已变化，请刷新轴列表后重新选择。";
    if (code === "character_trait_axis_not_found") return "所选作者轴不存在或不属于当前项目，请刷新轴列表。";
    if (code === "character_trait_axis_dimension_mismatch") return "所选作者轴不适用于这条核心性格候选，请重新选择。";
    if (code === "character_trait_axis_proposition_required") return "所选旧作者轴还没有正向命题，请先由作者补写并核对。";
    if (code === "character_trait_axis_proposition_conflict") return "作者轴正向命题已变化；原选择保留，请刷新轴列表并重新核对。";
    if (code === "character_trait_axis_alignment_required") return "当前候选还没有可核对的同向/反向映射；不确定时请保留待审。";
    if (code === "character_trait_axis_alignment_unverified") return "同轴旧特征的方向尚未补认；请先在角色档案逐条核对。";
    if (code === "character_trait_candidate_stale") return "来源文档或叙事上下文已变化；请重新分析后审核。";
    if (code === "character_trait_support_binding_invalid") return "精确证据定位无法与冻结原文核对；请重新分析后审核。";
    if (code === "character_trait_confirmation_conflict") return "同一作用域已有冲突的已确认特征；请核对角色档案与轴定义。";
    if (code === "character_trait_supersession_conflict") return "待替代的角色特征已变化；请刷新后重新核对。";
    if (code === "character_trait_revision_conflict") return "这条特征已在其他页面更新；档案正在刷新，请重新核对后再操作。";
    if (error.status === 409) return "档案或作者轴已在其他页面更新；请刷新后重新核对。";
    if (error.status === 429) return "请求过于频繁，请稍后重试。";
    if (error.status === 404) return "这项角色资料不存在，或不属于当前项目。";
    if (error.status === 401) return "登录状态已失效，请重新登录后继续。";
    if (error.status === 403) return "安全校验未通过，请刷新页面后重试。";
    if (error.status === 422) return "提交内容未通过校验，请检查轴名称、定义与候选状态。";
    return "请求没有完成，请稍后重试。";
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
  const scopeKey = reviewScopeKey(projectId, route.characterId, route.candidateId);
  const reviewScopeRef = useRef({ key: scopeKey, generation: 0 });
  reviewScopeRef.current = advanceReviewScope(reviewScopeRef.current, scopeKey);
  const [searchDraft, setSearchDraft] = useState(route.query);
  const [characterPage, setCharacterPage] = useState<CharacterPage | null>(null);
  const [characterDetail, setCharacterDetail] = useState<CharacterDetail | null>(null);
  const [candidatePage, setCandidatePage] = useState<CandidatePage | null>(null);
  const [withdrawnPage, setWithdrawnPage] = useState<CandidatePage | null>(null);
  const [withdrawnPageNumber, setWithdrawnPageNumber] = useState(1);
  const [withdrawnLoading, setWithdrawnLoading] = useState(false);
  const [withdrawnError, setWithdrawnError] = useState("");
  const [withdrawnReload, setWithdrawnReload] = useState(0);
  const [withdrawBusyId, setWithdrawBusyId] = useState<string | null>(null);
  const [withdrawError, setWithdrawError] = useState<{ id: string; message: string } | null>(null);
  const [candidate, setCandidate] = useState<ProfileCandidate | null>(null);
  const [neighborPage, setNeighborPage] = useState<SourceNeighborPage | null>(null);
  const [neighborLoading, setNeighborLoading] = useState(false);
  const [neighborMoreBusy, setNeighborMoreBusy] = useState(false);
  const [neighborError, setNeighborError] = useState("");
  const [neighborMoreError, setNeighborMoreError] = useState("");
  const [neighborReload, setNeighborReload] = useState(0);
  const [driftPage, setDriftPage] = useState<DriftIssuePage | null>(null);
  const [listLoading, setListLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [sectionLoading, setSectionLoading] = useState(false);
  const [candidateLoading, setCandidateLoading] = useState(false);
  const [decisionBusy, setDecisionBusy] = useState<CandidateDecision | null>(null);
  const [axisItems, setAxisItems] = useState<CharacterTraitAxis[]>([]);
  const [axisTotal, setAxisTotal] = useState(0);
  const [axisFetchedCount, setAxisFetchedCount] = useState(0);
  const [axisLoaded, setAxisLoaded] = useState(false);
  const [axisListDirty, setAxisListDirty] = useState(false);
  const [axisLoading, setAxisLoading] = useState(false);
  const [axisMoreBusy, setAxisMoreBusy] = useState(false);
  const [axisCreateBusy, setAxisCreateBusy] = useState(false);
  const [axisError, setAxisError] = useState("");
  const [axisMoreError, setAxisMoreError] = useState("");
  const [axisReload, setAxisReload] = useState(0);
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
  const axisMoreRequestRef = useRef(0);
  const neighborMoreRequestRef = useRef(0);
  const neighborMorePendingRef = useRef(false);
  const axisCreateRequestRef = useRef(0);
  const decisionRequestRef = useRef(0);
  const axisCreatePendingRef = useRef(false);
  const decisionPendingRef = useRef(false);
  const withdrawPendingRef = useRef(false);
  const withdrawRequestRef = useRef(0);
  const profileScopeKey = reviewScopeKey(projectId, route.characterId, null);
  const profileScopeRef = useRef({ key: profileScopeKey, generation: 0 });
  profileScopeRef.current = advanceReviewScope(profileScopeRef.current, profileScopeKey);

  useEffect(() => {
    // A previous candidate's mutation can still finish on the server. Its
    // loading and feedback must not follow the user into another scope.
    if (reviewScopeRef.current.key !== scopeKey) return;
    axisCreatePendingRef.current = false;
    decisionPendingRef.current = false;
    setAxisCreateBusy(false);
    setDecisionBusy(null);
    setActionError("");
    setAnnouncement("");
  }, [scopeKey]);

  useEffect(() => {
    if (profileScopeRef.current.key !== profileScopeKey) return;
    withdrawPendingRef.current = false;
    setWithdrawBusyId(null);
    setWithdrawError(null);
    setWithdrawnPageNumber(1);
  }, [profileScopeKey]);

  const prerequisiteReadiness: CharacterReadiness | null = projectId
    ? documentCount === 0
      ? "no_documents"
      : completedRunCount === 0
        ? "no_completed_run"
        : null
    : null;
  const readiness = prerequisiteReadiness || characterPage?.readiness || "ready";
  // A confirmed trait survives document retirement. A selected character is
  // also a direct detail target, independent of the roster search filter.
  const canBrowseCharacters = Boolean(characterPage?.total || route.characterId);
  const readinessView = canBrowseCharacters ? null : describeReadiness(readiness);

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
      projectLoading
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
    route.page,
    route.query,
    listReload,
  ]);

  useEffect(() => {
    setCharacterDetail(null);
    setDetailError("");
    if (!projectId || !route.characterId || !canBrowseCharacters) {
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
  }, [projectId, route.characterId, canBrowseCharacters, detailReload]);

  useEffect(() => {
    setWithdrawnPage(null);
    setWithdrawnError("");
    if (!projectId || !route.characterId || route.section !== "profile" || !canBrowseCharacters) {
      setWithdrawnLoading(false);
      return;
    }
    const controller = new AbortController();
    setWithdrawnLoading(true);
    void fetchWithdrawnProfileCandidates(
      projectId,
      route.characterId,
      { page: withdrawnPageNumber },
      controller.signal,
    )
      .then((page) => {
        if (!controller.signal.aborted) setWithdrawnPage(page);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        const message = requestError(error);
        if (message) setWithdrawnError(message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setWithdrawnLoading(false);
      });
    return () => controller.abort();
  }, [projectId, route.characterId, route.section, canBrowseCharacters, withdrawnPageNumber, withdrawnReload]);

  useEffect(() => {
    setCandidatePage(null);
    setDriftPage(null);
    setSectionError("");
    if (!projectId || !route.characterId || !canBrowseCharacters) {
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
    canBrowseCharacters,
    sectionReload,
  ]);

  useEffect(() => {
    setCandidate(null);
    setCandidateError("");
    if (
      !projectId ||
      !route.characterId ||
      route.section !== "candidates" ||
      !route.candidateId ||
      !canBrowseCharacters
    ) {
      setCandidateLoading(false);
      return;
    }
    const controller = new AbortController();
    const startedScope = reviewScopeRef.current;
    setCandidateLoading(true);
    void fetchProfileCandidate(
      projectId,
      route.characterId,
      route.candidateId,
      controller.signal,
    )
      .then((loaded) => {
        if (!controller.signal.aborted && isCurrentReviewScope(reviewScopeRef.current, startedScope)) {
          setCandidate(loaded);
        }
      })
      .catch((error) => {
        if (controller.signal.aborted || !isCurrentReviewScope(reviewScopeRef.current, startedScope)) return;
        const message = requestError(error);
        if (message) setCandidateError(message);
      })
      .finally(() => {
        if (!controller.signal.aborted && isCurrentReviewScope(reviewScopeRef.current, startedScope)) {
          setCandidateLoading(false);
        }
      });
    return () => controller.abort();
  }, [
    projectId,
    route.characterId,
    route.section,
    route.candidateId,
    canBrowseCharacters,
    candidateReload,
  ]);

  useEffect(() => {
    setAxisItems([]);
    setAxisTotal(0);
    setAxisFetchedCount(0);
    setAxisLoaded(false);
    setAxisListDirty(false);
    axisMoreRequestRef.current += 1;
    setAxisMoreBusy(false);
    setAxisError("");
    setAxisMoreError("");
    if (
      !projectId || !route.candidateId ||
      candidate?.id !== route.candidateId ||
      candidate.dimension !== "core_personality" ||
      !canBrowseCharacters
    ) {
      setAxisLoading(false);
      return;
    }
    const controller = new AbortController();
    const startedScope = reviewScopeRef.current;
    setAxisLoading(true);
    void fetchCharacterTraitAxes(projectId, { limit: 100, offset: 0 }, controller.signal)
      .then((page) => {
        if (controller.signal.aborted || !isCurrentReviewScope(reviewScopeRef.current, startedScope)) return;
        setAxisItems(page.items);
        setAxisTotal(page.total);
        setAxisFetchedCount(page.items.length);
        setAxisLoaded(true);
      })
      .catch((error) => {
        if (controller.signal.aborted || !isCurrentReviewScope(reviewScopeRef.current, startedScope)) return;
        const message = requestError(error);
        if (message) setAxisError(message);
      })
      .finally(() => {
        if (!controller.signal.aborted && isCurrentReviewScope(reviewScopeRef.current, startedScope)) {
          setAxisLoading(false);
        }
      });
    return () => controller.abort();
  }, [projectId, route.candidateId, candidate?.id, candidate?.dimension, canBrowseCharacters, axisReload]);

  useEffect(() => {
    setNeighborPage(null);
    setNeighborError("");
    setNeighborMoreError("");
    neighborMoreRequestRef.current += 1;
    neighborMorePendingRef.current = false;
    setNeighborMoreBusy(false);
    if (
      !projectId || !route.characterId || !route.candidateId ||
      route.section !== "candidates" || !canBrowseCharacters ||
      candidate?.id !== route.candidateId || !candidate.source_verified
    ) {
      setNeighborLoading(false);
      return;
    }
    const controller = new AbortController();
    const startedScope = reviewScopeRef.current;
    setNeighborLoading(true);
    void fetchSourceNeighbors(
      projectId, route.characterId, route.candidateId,
      { limit: 20, offset: 0 }, controller.signal,
    )
      .then((page) => {
        if (!controller.signal.aborted && isCurrentReviewScope(reviewScopeRef.current, startedScope)) {
          setNeighborPage(page);
        }
      })
      .catch((error) => {
        if (controller.signal.aborted || !isCurrentReviewScope(reviewScopeRef.current, startedScope)) return;
        setNeighborError(requestError(error));
      })
      .finally(() => {
        if (!controller.signal.aborted && isCurrentReviewScope(reviewScopeRef.current, startedScope)) {
          setNeighborLoading(false);
        }
      });
    return () => controller.abort();
  }, [projectId, route.characterId, route.candidateId, route.section, canBrowseCharacters, candidate?.id, candidate?.source_verified, neighborReload]);

  async function loadMoreNeighbors() {
    if (
      !route.characterId || !route.candidateId || !neighborPage ||
      !neighborPage.has_more || neighborLoading || neighborMorePendingRef.current
    ) return;
    const startedScope = reviewScopeRef.current;
    const requestId = ++neighborMoreRequestRef.current;
    neighborMorePendingRef.current = true;
    setNeighborMoreBusy(true);
    setNeighborMoreError("");
    try {
      const page = await fetchSourceNeighbors(
        projectId, route.characterId, route.candidateId,
        { limit: 20, offset: neighborPage.items.length },
      );
      if (!isCurrentReviewRequest(reviewScopeRef.current, startedScope, neighborMoreRequestRef.current, requestId)) return;
      const prior = neighborPage;
      const sameGroups = JSON.stringify(prior.source_groups) === JSON.stringify(page.source_groups);
      const ids = new Set(prior.items.map((item) => item.id));
      if (
        page.candidate_id !== prior.candidate_id ||
        page.source_run_id !== prior.source_run_id ||
        page.total !== prior.total || page.offset !== prior.items.length ||
        !sameGroups || page.items.some((item) => ids.has(item.id))
      ) {
        setNeighborMoreError("同源候选列表在翻页期间发生变化；请重新读取后核对。");
        return;
      }
      setNeighborPage({ ...page, items: [...prior.items, ...page.items] });
    } catch (error) {
      if (isCurrentReviewRequest(reviewScopeRef.current, startedScope, neighborMoreRequestRef.current, requestId)) {
        setNeighborMoreError(requestError(error));
      }
    } finally {
      if (isCurrentReviewRequest(reviewScopeRef.current, startedScope, neighborMoreRequestRef.current, requestId)) {
        neighborMorePendingRef.current = false;
        setNeighborMoreBusy(false);
      }
    }
  }

  async function loadMoreAxes() {
    if (
      axisLoading || axisMoreBusy || axisListDirty ||
      axisFetchedCount >= axisTotal || !route.candidateId
    ) return;
    const startedScope = reviewScopeRef.current;
    const requestId = ++axisMoreRequestRef.current;
    setAxisMoreBusy(true);
    setAxisMoreError("");
    try {
      const page = await fetchCharacterTraitAxes(projectId, { limit: 100, offset: axisFetchedCount });
      if (!isCurrentReviewRequest(reviewScopeRef.current, startedScope, axisMoreRequestRef.current, requestId)) return;
      let merged: CharacterTraitAxis[];
      try {
        merged = appendAxisPage(axisItems, page);
      } catch {
        setAxisLoaded(false);
        setAxisError("轴列表在分页读取期间发生变化；请重新读取，避免遗漏已有轴。");
        return;
      }
      setAxisItems(merged);
      setAxisFetchedCount((count) => count + page.items.length);
      setAxisTotal(page.total);
    } catch (error) {
      if (isCurrentReviewRequest(reviewScopeRef.current, startedScope, axisMoreRequestRef.current, requestId)) {
        setAxisMoreError(requestError(error));
      }
    } finally {
      if (isCurrentReviewRequest(reviewScopeRef.current, startedScope, axisMoreRequestRef.current, requestId)) {
        setAxisMoreBusy(false);
      }
    }
  }

  async function createAxis(input: { display_name: string; definition: string; positive_proposition: string }): Promise<CharacterTraitAxis> {
    if (
      !route.characterId || !route.candidateId ||
      candidate?.id !== route.candidateId ||
      candidate.character_id !== route.characterId ||
      candidate.dimension !== "core_personality" ||
      axisCreatePendingRef.current
    ) {
      throw new Error("请重新选择要审核的角色候选。");
    }
    const startedScope = reviewScopeRef.current;
    const requestId = ++axisCreateRequestRef.current;
    axisCreatePendingRef.current = true;
    setAxisCreateBusy(true);
    try {
      const created = await createCharacterTraitAxis(projectId, input);
      if (isCurrentReviewRequest(reviewScopeRef.current, startedScope, axisCreateRequestRef.current, requestId)) {
        setAxisItems((current) => current.some((item) => item.id === created.id)
          ? current : [...current, created]);
        // The response confirms this one new axis, not a stable pagination
        // snapshot. A parallel tab may have inserted another row meanwhile.
        setAxisListDirty(true);
      }
      return created;
    } finally {
      if (isCurrentReviewRequest(reviewScopeRef.current, startedScope, axisCreateRequestRef.current, requestId)) {
        axisCreatePendingRef.current = false;
        setAxisCreateBusy(false);
      }
    }
  }

  async function defineAxisProposition(axis: CharacterTraitAxis, positiveProposition: string): Promise<CharacterTraitAxis> {
    if (!route.characterId || !route.candidateId || candidate?.id !== route.candidateId || axis.positive_proposition) {
      throw new Error("请重新核对作者轴与当前候选。");
    }
    const startedScope = reviewScopeRef.current;
    const updated = await setAxisPositiveProposition(projectId, axis.id, {
      positive_proposition: positiveProposition,
      expected_axis_version: axis.version,
    });
    if (updated.id !== axis.id || !updated.positive_proposition_sha256) {
      throw new TypeError("作者轴命题保存结果不一致");
    }
    if (isCurrentReviewScope(reviewScopeRef.current, startedScope)) {
      setAxisItems((current) => current.map((item) => item.id === updated.id ? updated : item));
    }
    return updated;
  }

  function submitSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    navigate({
      query: searchDraft.trim().slice(0, 120),
      page: 1,
      characterId: null,
      candidateId: null,
    });
  }

  async function decide(
    decision: CandidateDecision,
    comment: string,
    selectedAxis: CharacterTraitAxis | null,
    alignment: "same" | "opposite" | null,
  ) {
    if (
      !candidate || !route.characterId || !route.candidateId ||
      candidate.id !== route.candidateId ||
      candidate.character_id !== route.characterId ||
      decisionBusy || decisionPendingRef.current
    ) return;
    if (decision === "confirm" && candidate.dimension === "core_personality" && !selectedAxis) {
      setActionError("请先为核心性格候选选择或创建作者轴。");
      return;
    }
    if (decision === "confirm" && candidate.dimension === "core_personality" &&
      (!selectedAxis?.positive_proposition_sha256 || !alignment)) {
      setActionError("请先核对作者轴正向命题，再明确选择同向或反向；不确定时保留待审。");
      return;
    }
    const startedScope = reviewScopeRef.current;
    const requestId = ++decisionRequestRef.current;
    const isCurrentRequest = () =>
      isCurrentReviewRequest(reviewScopeRef.current, startedScope, decisionRequestRef.current, requestId);
    decisionPendingRef.current = true;
    try {
      setDecisionBusy(decision);
      setActionError("");
      setAnnouncement("");
      await submitCandidateDecision(
        projectId,
        route.characterId,
        candidate.id,
        {
          decision,
          comment: comment.trim(),
          expected_revision: candidate.revision,
          ...(decision === "confirm" && selectedAxis
            ? {
                approved_axis_id: selectedAxis.id,
                expected_axis_version: selectedAxis.version,
                axis_alignment: alignment || undefined,
                expected_axis_positive_proposition_sha256: selectedAxis.positive_proposition_sha256 || undefined,
              }
            : {}),
        },
      );
      if (!isCurrentRequest()) return;
      // The mutation response is intentionally compact and does not carry
      // verified frozen evidence. Re-read the detail instead of replacing it
      // with a misleading unverified candidate, including on replay.
      setCandidate(null);
      setCandidateReload((value) => value + 1);
      setAnnouncement(
        decision === "confirm"
          ? "归纳已确认并写入角色档案；作者轴只影响之后创建的分析，既有报告不会改写。"
          : "归纳已驳回，决定已保留在审核记录中。",
      );
      setListReload((value) => value + 1);
      setDetailReload((value) => value + 1);
      setSectionReload((value) => value + 1);
      setNeighborReload((value) => value + 1);
    } catch (error) {
      if (!isCurrentRequest()) return;
      const message = requestError(error);
      setActionError(message);
      if (error instanceof ApiError && error.status === 409) {
        setCandidateReload((value) => value + 1);
        setDetailReload((value) => value + 1);
        if (
          error.detail && typeof error.detail === "object" &&
          "detail" in error.detail &&
          error.detail.detail && typeof error.detail.detail === "object" &&
          "code" in error.detail.detail &&
          [
            "character_trait_axis_version_conflict",
            "character_trait_axis_proposition_conflict",
            "character_trait_axis_proposition_required",
          ].includes(String(error.detail.detail.code))
        ) setAxisReload((value) => value + 1);
      }
    } finally {
      if (isCurrentRequest()) {
        decisionPendingRef.current = false;
        setDecisionBusy(null);
      }
    }
  }

  async function withdrawConfirmedTrait(item: CharacterProfileItem) {
    if (
      !route.characterId || item.revision === null ||
      withdrawPendingRef.current ||
      !characterDetail?.profile_items.some(
        (current) => current.id === item.id && current.revision === item.revision,
      )
    ) return;
    const characterId = route.characterId;
    const startedScope = profileScopeRef.current;
    const requestId = ++withdrawRequestRef.current;
    const isCurrentRequest = () =>
      isCurrentReviewRequest(profileScopeRef.current, startedScope, withdrawRequestRef.current, requestId);
    withdrawPendingRef.current = true;
    setWithdrawBusyId(item.id);
    setWithdrawError(null);
    setActionError("");
    setAnnouncement("");
    try {
      const result = await submitCandidateDecision(projectId, characterId, item.id, {
        decision: "withdraw",
        comment: "",
        expected_revision: item.revision,
      });
      if (!isCurrentRequest()) return;
      if (result.candidate.id !== item.id || result.candidate.status !== "withdrawn") {
        throw new TypeError("撤销响应与所选特征不一致");
      }
      setAnnouncement("特征已撤销，将不再参与之后新建的审查；既有运行报告不变。撤销记录可在档案下方查看。");
      setWithdrawnPageNumber(1);
      setWithdrawnReload((value) => value + 1);
      setListReload((value) => value + 1);
      setDetailReload((value) => value + 1);
    } catch (error) {
      if (!isCurrentRequest()) return;
      const message = requestError(error);
      setActionError(`撤销结果需要核对：${message}`);
      setWithdrawError({
        id: item.id,
        message: error instanceof ApiError
          ? message
          : `${message} 请刷新档案核对撤销是否已生效，再决定是否重试。`,
      });
      // A lost response does not prove the mutation failed. Re-read state
      // before offering the same action again, including on 409 conflicts.
      setListReload((value) => value + 1);
      setDetailReload((value) => value + 1);
      setWithdrawnReload((value) => value + 1);
    } finally {
      if (isCurrentRequest()) {
        withdrawPendingRef.current = false;
        setWithdrawBusyId(null);
      }
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

      {projectLoading || (listLoading && !characterPage) ? (
        <div className="characterWorkspaceLoading" aria-busy="true">正在读取项目资料…</div>
      ) : listError && !characterPage ? (
        <div className="characterPrerequisite characterPanelError" role="alert">
          <h2>角色档案列表没有加载</h2>
          <p>{listError}</p>
          <button type="button" onClick={() => setListReload((value) => value + 1)}>重试</button>
        </div>
      ) : documentCount > 0 && !baselineContextReady && !canBrowseCharacters ? (
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
                      <CharacterProfile
                        projectId={projectId}
                        characterId={route.characterId}
                        items={characterDetail.profile_items}
                        withdrawnPage={withdrawnPage}
                        withdrawnPageNumber={withdrawnPageNumber}
                        withdrawnLoading={withdrawnLoading}
                        withdrawnError={withdrawnError}
                        withdrawBusyId={withdrawBusyId}
                        withdrawError={withdrawError}
                        onWithdraw={(item) => void withdrawConfirmedTrait(item)}
                        onWithdrawnPage={setWithdrawnPageNumber}
                        onRetryWithdrawn={() => setWithdrawnReload((value) => value + 1)}
                        onAligned={() => {
                          setAnnouncement("作者轴方向已补认，仅影响之后新建的分析；历史报告保持不变。");
                          setDetailReload((value) => value + 1);
                          setListReload((value) => value + 1);
                        }}
                      />
                    )}
                    {route.section === "candidates" && (
                      <CandidateReview
                        key={reviewScopeKey(projectId, route.characterId, null)}
                        scopeKey={scopeKey}
                        page={candidatePage}
                        selected={
                          candidate?.id === route.candidateId &&
                          candidate.character_id === route.characterId
                            ? candidate : null
                        }
                        selectedId={route.candidateId}
                        neighborPage={neighborPage?.candidate_id === route.candidateId ? neighborPage : null}
                        neighborLoading={neighborLoading}
                        neighborMoreBusy={neighborMoreBusy}
                        neighborError={neighborError}
                        neighborMoreError={neighborMoreError}
                        loading={sectionLoading}
                        detailLoading={candidateLoading}
                        error={sectionError}
                        detailError={candidateError}
                        decisionBusy={decisionBusy}
                        axes={axisItems}
                        axisTotal={axisTotal}
                        axisFetchedCount={axisFetchedCount}
                        axisLoaded={axisLoaded}
                        axisListDirty={axisListDirty}
                        axisLoading={axisLoading}
                        axisMoreBusy={axisMoreBusy}
                        axisCreateBusy={axisCreateBusy}
                        axisError={axisError}
                        axisMoreError={axisMoreError}
                        axisRefreshKey={axisReload}
                        pageNumber={route.page}
                        hrefForCandidate={(candidateId) => pathFor({ ...route, candidateId })}
                        onSelect={(candidateId) => navigate({ candidateId })}
                        onPage={(page) => navigate({ page, candidateId: null })}
                        onRetry={() => setSectionReload((value) => value + 1)}
                        onRetryDetail={() => setCandidateReload((value) => value + 1)}
                        onRetryNeighbors={() => setNeighborReload((value) => value + 1)}
                        onLoadMoreNeighbors={() => void loadMoreNeighbors()}
                        onBack={() => navigate({ candidateId: null })}
                        onRetryAxes={() => setAxisReload((value) => value + 1)}
                        onLoadMoreAxes={() => void loadMoreAxes()}
                        onCreateAxis={createAxis}
                        onSetAxisProposition={defineAxisProposition}
                        onDecision={(decision, comment, axis, alignment) => void decide(decision, comment, axis, alignment)}
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
