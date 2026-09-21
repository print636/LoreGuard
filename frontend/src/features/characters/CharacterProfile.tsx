import {
  characterDimensionNames,
  profileOriginNames,
} from "./presentation";
import type {
  CharacterDimension,
  CharacterProfileItem,
} from "./types";

const dimensions = Object.keys(characterDimensionNames) as CharacterDimension[];

export default function CharacterProfile({
  items,
}: {
  items: CharacterProfileItem[];
}) {
  if (items.length === 0) {
    return (
      <div className="characterSectionEmpty">
        <h3>还没有已确认的角色档案</h3>
        <p>AI 归纳不会自动成为正式档案。请到“待确认归纳”逐条核对证据。</p>
      </div>
    );
  }

  return (
    <div className="characterProfileGroups">
      {dimensions.map((dimension) => {
        const rows = items.filter((item) => item.dimension === dimension);
        if (rows.length === 0) return null;
        return (
          <section key={dimension} aria-labelledby={`character-dimension-${dimension}`}>
            <div className="characterSubhead">
              <h3 id={`character-dimension-${dimension}`}>
                {characterDimensionNames[dimension]}
              </h3>
              <span>{rows.length} 条</span>
            </div>
            <ul>
              {rows.map((item) => (
                <li key={item.id}>
                  <p>{item.statement}</p>
                  <div>
                    <span>{profileOriginNames[item.origin]}</span>
                    <span>{item.evidence_count} 条证据</span>
                    {item.scopes.slice(0, 2).map((scope) => (
                      <span key={scope.scope_id}>{scope.label}</span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
