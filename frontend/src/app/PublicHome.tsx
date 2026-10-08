import { useEffect, useRef, useState } from "react";
import UserGuide from "../features/help/UserGuide";
import LoginReminder from "./LoginReminder";
import { publicEntryActions, publicSessionMessage, type PublicEntryAction, type PublicSessionStatus } from "./publicEntry";
import "./public-home.css";

type Props = {
  status: PublicSessionStatus;
  onRetry: () => void;
  draftLogoutWarning?: string;
};

type EntrySelection = { action: PublicEntryAction; trigger: HTMLButtonElement };

/** Public reading contains only original static copy and read-only local help. */
export default function PublicHome({ status, onRetry, draftLogoutWarning }: Props) {
  const [selection, setSelection] = useState<EntrySelection | null>(null);
  const [helpTrigger, setHelpTrigger] = useState<HTMLButtonElement | null>(null);
  const heading = useRef<HTMLHeadingElement | null>(null);

  useEffect(() => { heading.current?.focus({ preventScroll: true }); }, []);

  function entryButton(action: PublicEntryAction, primary = false) {
    return <button type="button" className={primary ? "quietPrimary" : undefined}
      onClick={(event) => setSelection({ action, trigger: event.currentTarget })}>{publicEntryActions[action].label}</button>;
  }

  return (
    <div className="publicHome productPage">
      <a className="skipLink" href="#public-home-main">跳到产品介绍</a>
      <header className="publicHomeHeader">
        <a className="brandLockup publicHomeBrand" href="/" aria-label="LoreGuard 公开首页">
          <span className="productSeal" aria-hidden="true"><i>LG</i></span>
          <span><b>LoreGuard</b><small>故事剧情一致性审查</small></span>
        </a>
        <nav aria-label="公开首页导航">
          <a href="#public-home-method">工作方式</a>
          <a href="#public-home-boundaries">判断边界</a>
          {entryButton("model")}
        </nav>
      </header>

      <main className="publicHomeMain" id="public-home-main">
        {draftLogoutWarning && <p className="publicHomeWarning" role="alert">{draftLogoutWarning}</p>}
        <section className="publicHomeWelcome" aria-labelledby="public-home-title">
          <div className="publicHomeWelcomeCopy">
            <div className="publicHomeReadingMark">
              <span className="emptyBookSpirit" aria-hidden="true"><i /><i /><b>··</b></span>
              <span>先了解，再打开你的故事</span>
            </div>
            <h1 id="public-home-title" ref={heading} tabIndex={-1}>让每一条判断<br />回到原文证据</h1>
            <p>LoreGuard 为作者与叙事设计者核对故事中的事实、世界规则和角色表现。直接从正文开始，也可以加入已确认的设定与旧稿，审查新增剧情。</p>
            <p>报告把判断与双方原文放在一起，帮助你决定哪里需要修订、哪里是合理的变化。</p>
            <div className="publicHomeWelcomeActions">
              {entryButton("workspace", true)}
              <button type="button" onClick={(event) => setHelpTrigger(event.currentTarget)}>阅读使用指南</button>
            </div>
            <p className="publicHomeSession" role="status">{publicSessionMessage(status)}</p>
            {status === "failed" && <button className="textButton" type="button" onClick={() => {
              onRetry();
              heading.current?.focus({ preventScroll: true });
            }}>重新确认登录状态</button>}
          </div>

          <section className="publicHomeExample" aria-labelledby="public-home-example-title">
            <div className="publicHomeExampleHeading">
              <h2 id="public-home-example-title">先看一组原文</h2>
              <span>原创静态示例</span>
            </div>
            <p>设定与新稿对照：潮汐门的开启条件</p>
            <div className="publicHomeEvidencePair">
              <div>
                <h3>已确认设定</h3>
                <p className="publicHomeEvidenceSource">《潮汐城记》设定 · 第 12 行</p>
                <blockquote>潮汐门只有在退潮钟响过三次后才能开启，涨潮时门轴会被海水锁住。</blockquote>
              </div>
              <div>
                <h3>待审新稿</h3>
                <p className="publicHomeEvidenceSource">《潮汐城记》第四章 · 第 28 行</p>
                <blockquote>涨潮钟刚响，闻舟便推开潮汐门，让满载药箱的小艇驶入内港。</blockquote>
              </div>
            </div>
            <div className="publicHomeExampleReading">
              <h3>需要作者核对</h3>
              <p>开启时机与既有规则可能冲突。先检查上下文是否交代了门轴改造、特殊许可或其他例外，再决定补写解释还是调整剧情。</p>
            </div>
            <p className="publicHomeExampleNote">这组摘录仅用于说明阅读方式，未经模型分析；浏览不会创建样例项目或消耗模型额度。</p>
          </section>
        </section>

        <section className="publicHomeSection" id="public-home-method" aria-labelledby="public-home-method-title">
          <div className="publicHomeSectionIntro">
            <h2 id="public-home-method-title">从文稿到可核对的报告</h2>
            <p>资料的身份、审查的范围和最后的判断，都由作者确认。</p>
          </div>
          <ol className="publicHomeMethod">
            <li><h3>准备故事资料</h3><p>导入正文，核对资料类型、发布状态与故事位置。只有新稿时，也可以进行有限范围的自检。</p></li>
            <li><h3>选择本次新稿</h3><p>确认作为依据的设定与历史；角色审查使用作者认可的特征基线，再主动开始校验。</p></li>
            <li><h3>对照证据再判断</h3><p>先看实际覆盖，再读问题说明与分析时原文。可以记录反馈，修订后自行发起复检。</p></li>
          </ol>
        </section>

        <section className="publicHomeSection" id="public-home-boundaries" aria-labelledby="public-home-boundaries-title">
          <div className="publicHomeSectionIntro">
            <h2 id="public-home-boundaries-title">它能提供依据，决定仍在作者</h2>
            <p>审查完成、模型可连接和故事没有问题，是三件不同的事。</p>
          </div>
          <ul className="publicHomeBoundaries">
            <li><h3>零问题不是准确性保证</h3><p>结果只描述本次输入与实际覆盖，仍可能有漏检。角色成长、伪装和情境变化需要结合上下文判断。</p></li>
            <li><h3>模型异常会明确说明</h3><p>超时或额度不足时，有限规则预检不能替代完整叙事理解。待复核线索与未验证提案也不等于正式问题。</p></li>
            <li><h3>文稿不会被自动改写或发布</h3><p>登录、导入和提交反馈都不会替你开始分析或定稿。使用模型可能产生费用，应在工作区核对配置和本次范围。</p></li>
          </ul>
        </section>

        <section className="publicHomeStart" aria-labelledby="public-home-start-title">
          <div><h2 id="public-home-start-title">准备好自己的故事时</h2><p>登录后进入项目中心，再选择你的起步方式。</p></div>
          <div className="publicHomeStartActions">{entryButton("create")}{entryButton("import")}</div>
        </section>
      </main>

      <footer className="publicHomeFooter">阅读本页和使用指南无需账户。个人文稿、项目记录与模型密钥仅在登录后的工作区操作。</footer>
      {selection && <LoginReminder {...selection} status={status} onRetry={onRetry} onClose={() => setSelection(null)} />}
      {helpTrigger && <UserGuide initialTopic="start" trigger={helpTrigger} onClose={() => setHelpTrigger(null)} />}
    </div>
  );
}
