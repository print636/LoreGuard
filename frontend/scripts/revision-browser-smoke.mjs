import { chromium } from "@playwright/test";

function required(name) {
  const prefix = `--${name}=`;
  const argument = process.argv.find((value) => value.startsWith(prefix));
  if (!argument || !argument.slice(prefix.length)) {
    throw new Error(`missing ${prefix}<value>`);
  }
  return argument.slice(prefix.length);
}

function optional(name) {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
}

const baseUrl = required("base-url").replace(/\/$/, "");
const projectId = required("project-id");
const baselineRunId = required("baseline-run-id");
const targetRunId = required("target-run-id");
const otherProjectId = required("other-project-id");
const otherBaselineRunId = required("other-baseline-run-id");
const screenshotPath = required("screenshot");
const browserExecutable = optional("browser-executable");
const expectedOutcomes = (optional("expected-outcomes") || "persisting,unverifiable")
  .split(",")
  .map((value) => value.trim())
  .filter(Boolean);

function revisionUrl(project, baseline, target) {
  const query = new URLSearchParams({
    step: "compare",
    outcome: "all",
    recheck: target,
  });
  return `${baseUrl}/app/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(baseline)}/revise?${query}`;
}

async function hasHorizontalOverflow(page) {
  return page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth,
  );
}

const browser = await chromium.launch({
  headless: true,
  ...(browserExecutable ? { executablePath: browserExecutable } : {}),
});
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
const result = {};

try {
  const correctUrl = revisionUrl(projectId, baselineRunId, targetRunId);
  const wrongUrl = revisionUrl(otherProjectId, otherBaselineRunId, targetRunId);

  await page.goto(correctUrl, { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "修订与复检" }).waitFor();
  for (const outcome of expectedOutcomes) {
    await page.locator(`.comparisonList article.${outcome}`).first().waitFor();
    result[`outcome_${outcome}`] = true;
  }
  result.desktopNoOverflow = !(await hasHorizontalOverflow(page));
  result.honestCopyVisible = await page.getByText("本次未再检出", { exact: true }).first().isVisible();
  await page.screenshot({ path: screenshotPath, fullPage: true });

  await page.setViewportSize({ width: 360, height: 800 });
  await page.reload({ waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "修订与复检" }).waitFor();
  result.mobileNoOverflow = !(await hasHorizontalOverflow(page));

  await page.goto(wrongUrl, { waitUntil: "networkidle" });
  await page.getByText(/服务器返回的比较结果不属于当前项目/).waitFor();
  result.foreignLineageRejected = true;
  result.foreignEvidenceHidden = (await page.locator(".comparisonList article").count()) === 0;

  await page.goBack({ waitUntil: "networkidle" });
  for (const outcome of expectedOutcomes) {
    await page.locator(`.comparisonList article.${outcome}`).first().waitFor();
  }
  result.backRestoresExactComparison = true;

  await page.goForward({ waitUntil: "networkidle" });
  await page.getByText(/服务器返回的比较结果不属于当前项目/).waitFor();
  result.forwardStillFailsClosed = true;

  if (Object.values(result).some((value) => value !== true)) {
    throw new Error(`revision browser smoke failed: ${JSON.stringify(result)}`);
  }
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} finally {
  await browser.close();
}
