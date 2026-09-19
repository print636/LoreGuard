# LoreGuard Design System — Master

> Global source of truth for LoreGuard product UI. Before changing a page, check
> `design-system/loreguard/pages/<page>.md`; a page file may override this file only
> for that page. Do not create a second global token source.

**Project:** LoreGuard

**Established direction:** 星轨虚拟书斋 / dark creator workspace

**Generated:** 2026-09-19

**Stack:** React 19 + TypeScript + Vite 8 + plain CSS
**Design dials:** variance 6/10 · motion 3/10 · density 7/10

## Design intent

LoreGuard is a professional review workspace for authors and narrative designers,
not a generic AI chatbot and not a decorative game launcher. Its interface should
feel like a quiet night-time study whose tools happen to be unusually capable.

- Keep the established deep-indigo, lavender, cyan and pink identity.
- Spend visual intensity on one element: the prismatic **开始校验** action.
- Keep long-form reading and evidence comparison calm, flat and high contrast.
- Use the book spirit only on welcome, empty and help states. It must never compete
  with a report, editor or running task.
- Prefer one continuous work surface, structured lists and dividers over a sea of
  identical rounded cards.
- AI activity is explicit: show what is running, what data it uses, whether the
  model is available, and how a failure can be recovered.

The initial UI/UX Pro Max query returned a generic light SaaS system. That palette
and marketing-page pattern were rejected because they conflict with the existing
dark creator workspace and the user's confirmed art direction. Applicable guidance
retained here: restrained motion, semantic tokens, visible focus, deep links,
accessible authentication, recoverable errors and route-level code splitting.

## Color tokens

All feature CSS consumes semantic variables. Raw colors belong only in the token
definition layer.

| Token | Value | Role |
|---|---:|---|
| `--color-page-deep` | `#090716` | Browser canvas, deepest background |
| `--color-page` | `#0D0A1E` | Application background |
| `--color-surface` | `#1D1739` | Main work surface |
| `--color-surface-raised` | `#291F4B` | Popover, modal, selected panel |
| `--color-field` | `#100D24` | Inputs and evidence/code wells |
| `--color-text` | `#F4F0FF` | Primary copy |
| `--color-text-secondary` | `#C7BEDD` | Supporting copy |
| `--color-text-muted` | `#A79EBC` | Metadata; do not use smaller than 12px |
| `--color-border` | `rgba(179,154,255,.20)` | Passive divider |
| `--color-border-strong` | `rgba(183,161,255,.38)` | Field/control boundary |
| `--color-lavender` | `#B7A2FF` | Brand and selected state |
| `--color-cyan` | `#8BE9F0` | Focus, links and running state |
| `--color-pink` | `#F2A7D5` | Sparing narrative accent |
| `--color-danger` | `#FF7FAD` | Confirmed problem / destructive action |
| `--color-warning` | `#FFBD82` | Needs attention / degraded state |
| `--color-success` | `#8DE5B8` | Completed / connected |
| `--color-focus` | `#8BE9F0` | Keyboard focus ring |

### Prismatic scan action

`linear-gradient(110deg, #F266D8 0%, #A876FF 48%, #69E8F2 100%)`
with `#100B24` text. This treatment is reserved for **开始校验** and its running
state. New project, save, import and confirm actions use a quiet solid lavender or
neutral treatment; they must not copy the gradient.

Status meaning always includes text or an icon; color is never the only signal.
Normal text must meet WCAG AA 4.5:1, non-text boundaries and focus indicators 3:1.

## Typography

- **UI/body:** `"Noto Sans SC", "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif`.
- **Narrative display:** `"Noto Serif SC", "Source Han Serif SC", "STKaiti", serif`.
  Use only for product name, welcome headings and short section titles—not tables,
  forms, metrics or long report copy.
- **Identifiers/data:** `"Cascadia Code", "SFMono-Regular", monospace`; only for
  IDs, timings, tokens and code-like evidence coordinates.
- Body: 16px/1.6 on mobile, 15–16px/1.6 on desktop. Never put essential copy below
  12px. Long-form lines target 60–75 Chinese/Latin character units.
- Type scale: 12 metadata · 14 secondary · 16 body · 18 title · 24 section ·
  32 page title · 44 welcome display.
- Sentence case is the default. Avoid decorative uppercase eyebrows; technical
  acronyms may remain uppercase when they carry real meaning.

## Spacing, shape and elevation

| Token | Value | Usage |
|---|---:|---|
| `--space-1` | `4px` | Tight internal alignment |
| `--space-2` | `8px` | Icon/label gap |
| `--space-3` | `12px` | Compact controls |
| `--space-4` | `16px` | Default component spacing |
| `--space-5` | `24px` | Section padding |
| `--space-6` | `32px` | Major separation |
| `--space-7` | `48px` | Welcome/empty-state breathing room |

- Radius scale: 8px controls · 12px grouped content · 16px major panel · 20px
  application shell. Pills are restricted to tags, statuses and compact counters.
- Elevation: borders and surface contrast first; shadows only for popovers, modals
  and the active scan action. Decorative blur does not define hierarchy.
- Prefer flat rows with dividers for project, document, run and issue collections.
  Use cards only for selectable objects or deliberately grouped summaries.
- Web pointer targets are at least 24×24 CSS px; primary and mobile controls target
  44px height with at least 8px separation.

## Layout system

- Breakpoints: 375 / 768 / 1024 / 1440px, with 360px as the hard minimum test.
- Desktop (≥1024): persistent 224px sidebar, flexible main canvas, optional
  320–400px context rail only where it materially helps.
- Tablet (768–1023): collapsible sidebar/drawer; contextual rail becomes a tab or
  drawer, never a permanently compressed third column.
- Mobile (<768): top app bar plus up to five labelled destinations; one main scroll
  region; sticky bottom scan action only on the scan page.
- Desktop content max-width: 1440px for dashboards, 1120px for forms/settings,
  960px for readable reports. Evidence comparison may use the full available width.
- Avoid nested page scrolling. Fixed/sticky controls reserve their own space and
  cannot obscure keyboard focus.

## Navigation and routing

- Every meaningful view has a URL; browser back/forward restores page, filters,
  selected run and scroll position where practical.
- Desktop navigation always includes icon + label and a visible active state.
- After a route change, focus the main heading or main region without stealing
  focus during in-page updates.
- Use breadcrumbs only at three or more hierarchy levels, for example
  `项目 / 星门纪事 / 运行 #128`.
- Provider configuration belongs to settings, not the project's primary workflow.
- Destructive account actions and logout are separated from ordinary navigation.

## Core component rules

### Buttons

- One primary action per view. Only **开始校验** receives the prismatic treatment.
- Async buttons disable duplicate submission and keep a stable width while showing
  progress. Copy names the result: `创建项目`, `导入文稿`, `保存设置`.
- Icon-only controls require an accessible name and at least a 44px hit area on
  touch layouts. Use one outline SVG family; no emoji as structural icons.
- Visible `:focus-visible`: 2px cyan ring plus 2px offset against the local surface.

### Forms and authentication

- Visible labels, persistent helper text for non-obvious fields and errors directly
  below the field via `aria-describedby`.
- Validate on blur; on failed submit retain inline errors and move focus to a linked
  error summary when multiple fields fail.
- Password inputs allow paste, autofill and password managers and provide a
  show/hide control. Never require memory puzzles or block paste.
- Long editor forms auto-save drafts; leaving with unsaved data requires explicit
  confirmation.

### Feedback states

- <300ms: no spinner. 300ms–1s: button-local progress. >1s: skeleton or stable
  progress region. Long analysis uses named SSE stages, percentage and cancel.
- Toasts confirm non-critical completion and never replace field errors. Use
  `aria-live="polite"`, keep them dismissible and auto-dismiss after 3–5s.
- Errors state cause + consequence + next action. Provider failure never erases the
  deterministic report; say that the run degraded and offer retry/configuration.
- Empty states distinguish `not created`, `not run`, `zero issues`, and `not
  available`; they must not all say “暂无数据”.

### Motion

- Shared timings: 120ms press · 180ms hover/focus · 240ms drawer/modal enter ·
  160ms exit. Use transform/opacity only.
- At most one ambient motion area per welcome/empty view. No background animation
  in report reading mode. All motion is interruptible and disabled or reduced by
  `prefers-reduced-motion`.

## Content voice

Plain, calm and evidence-led. Explain system boundaries without developer jargon.

- Good: `模型连接超时。本次分析已使用本地规则完成，你可以检查连接后重试。`
- Bad: `ProviderRetryExhausted: chunk=0` as the only user-facing message.
- Good empty state: `还没有项目。导入现有文稿，或从原创样例了解一次完整校验。`
- Do not anthropomorphize failures or imply the model has certainty it does not.

## Forbidden patterns

- Generic marketing hero inside the authenticated product.
- A grid of identical cards for every object and metric.
- Rainbow gradient on actions other than **开始校验**.
- Book spirit, particles or glow behind dense reports and editors.
- Placeholder-only labels, hover-only actions, hidden focus rings or color-only
  severity.
- Static single URL with state-only view switching.
- Model keys in `localStorage`, query strings, analytics or client logs.
- Blocking full-page spinner for long analysis.
- Emoji used as navigation or control icons.

## Delivery checklist

- [ ] Routes deep-link and browser back/forward behave predictably.
- [ ] 360, 375, 768, 1024 and 1440px have no horizontal page scroll.
- [ ] Keyboard-only operation reaches every action in logical order.
- [ ] Focus is visible and not obscured by sticky UI.
- [ ] Authentication supports paste, autofill and password managers.
- [ ] Loading, empty, error, offline, session-expired and permission states exist.
- [ ] Text contrast is ≥4.5:1 and UI boundaries/focus ≥3:1.
- [ ] Color is not the only status/severity signal.
- [ ] Touch actions are at least 44px high where used on mobile.
- [ ] Reduced motion leaves content immediately usable.
- [ ] Long lists (50+ rows) are paginated or virtualized.
- [ ] Route-level lazy loading keeps graph/timeline code out of the initial bundle.
- [ ] No secret, prompt, story text or model response is written to client logs.
- [ ] Book spirit appears only in welcome, empty or help contexts.
- [ ] Prismatic styling appears only on **开始校验**.
