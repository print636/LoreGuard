import type { WorkspaceView } from "../../routing";

const groups: Array<{ title: string; items: Array<[WorkspaceView, string, string, string]> }> = [
  { title: "故事资料", items: [
    ["projects", "文稿与资料", "项目与文档", "M4 4h12v15H4zM8 4v15m8-11h4v14H8v-3"],
    ["characters", "角色档案", "角色档案", "M8 7a4 4 0 1 0 8 0 4 4 0 0 0-8 0M4 21v-2a8 8 0 0 1 16 0v2"],
  ] },
  { title: "审查与修订", items: [
    ["check", "准备审查", "文稿校验", "M6 3h9l4 4v14H6zM15 3v5h4M9 14l2 2 5-5"],
    ["report", "审查报告", "完整报告", "M5 3h10l4 4v14H5zM15 3v5h4M8 12h8m-8 4h6"],
    ["revision", "修订复检", "修订复检", "M4 5h10v14H4zM15 9l3-3 3 3m-3-3v10a4 4 0 0 1-4 4"],
    ["diff", "版本对比", "版本对比", "M4 3h6v18H4zM14 3h6v18h-6zM6 8h2m8 6h2"],
  ] },
  { title: "查阅工具", items: [
    ["audit", "任务与记录", "运行审计", "M12 3a9 9 0 1 0 9 9M12 7v5l3 2M16 3h5v5"],
    ["visual", "关系与时间", "关系与时间", "M5 12h14M7 6l10 12M7 18 17 6M3 10h4v4H3zM17 10h4v4h-4zM5 4h4v4H5zM15 16h4v4h-4z"],
  ] },
];

export default function StoryNavigation({ view, onNavigate, busy }: {
  view: WorkspaceView; onNavigate: (view: WorkspaceView) => void; busy: boolean;
}) {
  return <nav className="storyNavigation" aria-label="作品内导航">
    {groups.map(group => <div className="storyNavGroup" key={group.title}>
      <p>{group.title}</p>
      {group.items.map(([id, label, accessibleLabel, path]) => <button key={id} type="button"
        className={view === id ? "active" : ""} aria-current={view === id ? "page" : undefined}
        aria-label={accessibleLabel} title={label} onClick={() => onNavigate(id)}>
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d={path} /></svg>
        <span>{label}</span>{id === "audit" && busy && <small>运行中</small>}
      </button>)}
    </div>)}
  </nav>;
}
