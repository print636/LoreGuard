import { type FormEvent, useEffect, useRef } from "react";
import type { CharacterSummary } from "./types";

type CharacterRosterProps = {
  characters: CharacterSummary[];
  selectedId: string | null;
  query: string;
  searchDraft: string;
  page: number;
  total: number;
  hasMore: boolean;
  loading: boolean;
  error: string;
  onSearchDraftChange: (value: string) => void;
  onSearch: (event: FormEvent<HTMLFormElement>) => void;
  onClearSearch: () => void;
  onSelect: (characterId: string) => void;
  hrefFor: (characterId: string) => string;
  onPage: (page: number) => void;
  onRetry: () => void;
};

export default function CharacterRoster({
  characters,
  selectedId,
  query,
  searchDraft,
  page,
  total,
  hasMore,
  loading,
  error,
  onSearchDraftChange,
  onSearch,
  onClearSearch,
  onSelect,
  hrefFor,
  onPage,
  onRetry,
}: CharacterRosterProps) {
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const characterLinkRefs = useRef(new Map<string, HTMLAnchorElement>());
  const previousSelectedIdRef = useRef<string | null>(selectedId);

  useEffect(() => {
    const previousId = previousSelectedIdRef.current;
    previousSelectedIdRef.current = selectedId;
    if (!previousId || selectedId) return;
    requestAnimationFrame(() => {
      (characterLinkRefs.current.get(previousId) || searchInputRef.current)?.focus();
    });
  }, [selectedId]);

  return (
    <aside
      className={`characterRoster ${selectedId ? "mobileRosterHidden" : ""}`}
      aria-label="角色列表"
      aria-busy={loading}
    >
      <form className="characterSearch" role="search" onSubmit={onSearch}>
        <label htmlFor="character-search">搜索角色</label>
        <div>
          <input
            ref={searchInputRef}
            id="character-search"
            type="search"
            maxLength={120}
            value={searchDraft}
            onChange={(event) => onSearchDraftChange(event.target.value)}
            placeholder="姓名或别名"
          />
          <button type="submit">搜索</button>
        </div>
        {query && (
          <button className="characterClearSearch" type="button" onClick={onClearSearch}>
            清除“{query}”
          </button>
        )}
      </form>

      <div className="characterRosterCount">
        <b>{total}</b>
        <span>位角色</span>
      </div>

      {error ? (
        <div className="characterPanelError" role="alert">
          <b>角色列表没有加载</b>
          <p>{error}</p>
          <button type="button" onClick={onRetry}>重试</button>
        </div>
      ) : loading ? (
        <div className="characterRosterSkeleton" aria-label="正在加载角色">
          <i /><i /><i /><i />
        </div>
      ) : characters.length === 0 ? (
        <div className="characterFilteredEmpty">
          <b>{query ? "没有匹配的角色" : "没有可显示的角色"}</b>
          <p>
            {query
              ? "换一个姓名或别名，或清除搜索条件。"
              : "这不代表资料中没有角色，请核对归纳覆盖范围。"}
          </p>
          {query && <button type="button" onClick={onClearSearch}>清除搜索</button>}
        </div>
      ) : (
        <ul className="characterRosterList">
          {characters.map((character) => (
            <li key={character.id}>
              <a
                ref={(node) => {
                  if (node) characterLinkRefs.current.set(character.id, node);
                  else characterLinkRefs.current.delete(character.id);
                }}
                href={hrefFor(character.id)}
                aria-current={selectedId === character.id ? "page" : undefined}
                onClick={(event) => {
                  if (
                    event.button !== 0 ||
                    event.metaKey ||
                    event.ctrlKey ||
                    event.shiftKey ||
                    event.altKey
                  ) return;
                  event.preventDefault();
                  onSelect(character.id);
                }}
              >
                <span className="characterAvatar" aria-hidden="true">
                  {character.canonical_name.slice(0, 1)}
                </span>
                <span className="characterRosterIdentity">
                  <b>{character.canonical_name}</b>
                  <small>
                    {character.aliases.length
                      ? character.aliases.slice(0, 2).join("、")
                      : "暂无别名"}
                  </small>
                </span>
                <span className="characterRosterSignals">
                  <small>{character.confirmed_item_count} 已确认</small>
                  {character.pending_candidate_count > 0 && (
                    <strong>{character.pending_candidate_count} 待确认</strong>
                  )}
                  {character.drift_issue_count !== null &&
                    character.drift_issue_count > 0 && (
                    <em>{character.drift_issue_count} 漂移</em>
                  )}
                </span>
              </a>
            </li>
          ))}
        </ul>
      )}

      {(page > 1 || hasMore) && (
        <nav className="characterPagination" aria-label="角色分页">
          <button type="button" disabled={page <= 1 || loading} onClick={() => onPage(page - 1)}>
            上一页
          </button>
          <span>第 {page} 页</span>
          <button type="button" disabled={!hasMore || loading} onClick={() => onPage(page + 1)}>
            下一页
          </button>
        </nav>
      )}
    </aside>
  );
}
