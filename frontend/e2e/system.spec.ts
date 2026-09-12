import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { expect, test, type FileChooser, type Page } from "@playwright/test";

const repositoryRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
const composeProject = process.env.LOREGUARD_E2E_COMPOSE_PROJECT || "loreguard-e2e";
const composeArguments = [
  "compose",
  "--project-name",
  composeProject,
  "--env-file",
  path.join(repositoryRoot, "tests/system/compose.env"),
  "-f",
  path.join(repositoryRoot, "docker-compose.yml"),
  "-f",
  path.join(repositoryRoot, "docker-compose.e2e.yml"),
];

const worldDocx = Buffer.from(
  "UEsDBBQAAAAIAPhsLF15bjPX6AAAAK0BAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbH1QyU7DMBD9FWuuKHHggBCK0wPLETiUDxjZk8SqN3nc0v49Tlt6QIXjzFv1+tXeO7GjzDYGBbdtB4KCjsaGScHn+rV5AMEFg0EXAyk4EMNq6NeHRCyqNrCCuZT0KCXrmTxyGxOFiowxeyz1zJNMqDc4kbzrunupYygUSlMWDxj6Zxpx64p42df3qUcmxyCeTsQlSwGm5KzGUnG5C+ZXSnNOaKvyyOHZJr6pBJBXExbk74Cz7r0Ok60h8YG5vKGvLPkVs5Em6q2vyvZ/mys94zhaTRf94pZy1MRcF/euvSAebfjpL49zD99QSwMEFAAAAAgA+GwsXZv9N+qtAAAAKQEAAAsAAABfcmVscy8ucmVsc43POw7CMAwG4KtE3mlaBoRQ0y4IqSsqB7ASN61oHkrCo7cnAwNFDIy2f3+W6/ZpZnanECdnBVRFCYysdGqyWsClP232wGJCq3B2lgQsFKFt6jPNmPJKHCcfWTZsFDCm5A+cRzmSwVg4TzZPBhcMplwGzT3KK2ri27Lc8fBpwNpknRIQOlUB6xdP/9huGCZJRydvhmz6ceIrkWUMmpKAhwuKq3e7yCzwpuarF5sXUEsDBBQAAAAIAPhsLF2E1iqv3wAAAB8BAAARAAAAd29yZC9kb2N1bWVudC54bWyzsa/IzVEoSy0qzszPs1Uy1DNQUkjNS85PycxLt1UKDXHTtVBSKC5JzEtJzMnPS7VVqkwtVrK3sym3SslPLs1NzStRABqQV2xVbquUUVJSYKWvX5yckZqbWKyXX5CaB5RLyy/KTSwBcovS9cvzi1IKivKTU4uLgebn5ugbGRiY6ecmZuYpgYxMyk+pBNEFIKIIRJTYPdkx7fnUnhfLm16s2/d03SwbfZAgiCwCkwXo6p/Nm/5sX8fzWS1P+ye+6Nz0bMb6l5O3ARmPG5qw6i1OTS4JKNIHC0Ds10f4zQ4AUEsBAhQAFAAAAAgA+GwsXXluM9foAAAArQEAABMAAAAAAAAAAAAAAIABAAAAAFtDb250ZW50X1R5cGVzXS54bWxQSwECFAAUAAAACAD4bCxdm/036q0AAAApAQAACwAAAAAAAAAAAAAAgAEZAQAAX3JlbHMvLnJlbHNQSwECFAAUAAAACAD4bCxdhNYqr98AAAAfAQAAEQAAAAAAAAAAAAAAgAHvAQAAd29yZC9kb2N1bWVudC54bWxQSwUGAAAAAAMAAwC5AAAA/QIAAAAA",
  "base64",
);
const chapterDocx = Buffer.from(
  "UEsDBBQAAAAIAPhsLF15bjPX6AAAAK0BAAATAAAAW0NvbnRlbnRfVHlwZXNdLnhtbH1QyU7DMBD9FWuuKHHggBCK0wPLETiUDxjZk8SqN3nc0v49Tlt6QIXjzFv1+tXeO7GjzDYGBbdtB4KCjsaGScHn+rV5AMEFg0EXAyk4EMNq6NeHRCyqNrCCuZT0KCXrmTxyGxOFiowxeyz1zJNMqDc4kbzrunupYygUSlMWDxj6Zxpx64p42df3qUcmxyCeTsQlSwGm5KzGUnG5C+ZXSnNOaKvyyOHZJr6pBJBXExbk74Cz7r0Ok60h8YG5vKGvLPkVs5Em6q2vyvZ/mys94zhaTRf94pZy1MRcF/euvSAebfjpL49zD99QSwMEFAAAAAgA+GwsXZv9N+qtAAAAKQEAAAsAAABfcmVscy8ucmVsc43POw7CMAwG4KtE3mlaBoRQ0y4IqSsqB7ASN61oHkrCo7cnAwNFDIy2f3+W6/ZpZnanECdnBVRFCYysdGqyWsClP232wGJCq3B2lgQsFKFt6jPNmPJKHCcfWTZsFDCm5A+cRzmSwVg4TzZPBhcMplwGzT3KK2ri27Lc8fBpwNpknRIQOlUB6xdP/9huGCZJRydvhmz6ceIrkWUMmpKAhwuKq3e7yCzwpuarF5sXUEsDBBQAAAAIAPhsLF3WHDAL1wAAABkBAAARAAAAd29yZC9kb2N1bWVudC54bWyzsa/IzVEoSy0qzszPs1Uy1DNQUkjNS85PycxLt1UKDXHTtVBSKC5JzEtJzMnPS7VVqkwtVrK3sym3SslPLs1NzStRABqQV2xVbquUUVJSYKWvX5yckZqbWKyXX5CaB5RLyy/KTSwBcovS9cvzi1IKivKTU4uLgebn5ugbGRiY6ecmZuYpgYxMyk+pBNEFIKIIRJTYPV+z5smOhuerF9jog7ggsghMFqCrfDZv+rN9Hc9ntTztn/iic9OzGetf7gYxHjc0YdVbnJpcElCkDxaA2KyP8JUdAFBLAQIUABQAAAAIAPhsLF15bjPX6AAAAK0BAAATAAAAAAAAAAAAAACAAQAAAABbQ29udGVudF9UeXBlc10ueG1sUEsBAhQAFAAAAAgA+GwsXZv9N+qtAAAAKQEAAAsAAAAAAAAAAAAAAIABGQEAAF9yZWxzLy5yZWxzUEsBAhQAFAAAAAgA+GwsXdYcMAvXAAAAGQEAABEAAAAAAAAAAAAAAIAB7wEAAHdvcmQvZG9jdW1lbnQueG1sUEsFBgAAAAADAAMAuQAAAPUCAAAAAA==",
  "base64",
);

function compose(...args: string[]) {
  execFileSync("docker", [...composeArguments, ...args], {
    cwd: repositoryRoot,
    stdio: "pipe",
  });
}

async function chooseDocx(
  page: Page,
  name: string,
  buffer: Buffer,
): Promise<void> {
  const chooserPromise = page.waitForEvent("filechooser");
  await page.getByLabel(/\u9009\u62e9.*\u6587\u4ef6/).click();
  const chooser: FileChooser = await chooserPromise;
  await chooser.setFiles({
    name,
    mimeType:
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    buffer,
  });
  await expect(page.locator(".uploadSelection")).toContainText(name);
}

test("Nginx 入口串联 DOCX 上传、排队恢复与证据报告", async ({ page }) => {
  const projectName = `Playwright 接线恢复验收 ${Date.now()}`;
  const browserApiOrigins = new Set<string>();
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname.startsWith("/api/") || url.pathname === "/health") {
      browserApiOrigins.add(url.origin);
    }
  });

  await page.goto("/");
  expect(new URL(page.url()).origin).toBe("http://127.0.0.1:8080");
  await page.getByRole("button", { name: "项目与文档" }).click();

  const projectControls = page.locator(".projectControls");
  await projectControls.getByPlaceholder("新项目名称").fill(projectName);
  await projectControls
    .getByRole("button", { name: "新建项目", exact: true })
    .click();
  await expect(page.getByText(`当前：${projectName}`, { exact: false })).toBeVisible();

  await page.getByLabel("上传文档类型").selectOption("canon");
  await chooseDocx(page, "world.docx", worldDocx);
  await page.getByRole("button", { name: "上传 1 个文件" }).click();
  await expect(page.getByRole("cell", { name: "world.docx" })).toBeVisible();

  await page.getByLabel("上传文档类型").selectOption("chapter");
  await chooseDocx(page, "chapter.docx", chapterDocx);
  await page.getByRole("button", { name: "上传 1 个文件" }).click();
  await expect(page.getByRole("cell", { name: "chapter.docx" })).toBeVisible();

  let workerPaused = false;
  let eventRequests = 0;
  await page.route("**/api/v1/analysis-runs/*/events", async (route) => {
    eventRequests += 1;
    if (eventRequests === 1) await route.abort("connectionrefused");
    else await route.continue();
  });

  try {
    compose("pause", "worker");
    workerPaused = true;
    await projectControls
      .getByRole("button", { name: /分析当前项目/ })
      .click();

    await expect(page.locator(".railLive")).toContainText("正在校验");
    await expect(page.locator(".railLive")).toContainText(
      "连接暂时中断，正在自动重连",
    );
    await expect.poll(() => eventRequests).toBeGreaterThanOrEqual(2);

    await page.reload();
    await page.getByRole("button", { name: "项目与文档" }).click();
    const projectSelect = page.getByLabel("选择项目");
    const projectValue = await projectSelect
      .locator("option")
      .filter({ hasText: projectName })
      .getAttribute("value");
    expect(projectValue).toBeTruthy();
    await projectSelect.selectOption(projectValue!);
    await expect(page.getByText(`当前：${projectName}`, { exact: false })).toBeVisible();
    await page.getByRole("button", { name: "运行审计" }).first().click();
    await expect(page.locator(".tables .badge.queued")).toBeVisible();
    await page.locator(".tables").getByRole("button", { name: "恢复" }).click();
    await expect(page.locator(".railLive")).toContainText("正在校验");

    compose("unpause", "worker");
    workerPaused = false;
    await expect(page.locator(".railLiveTop span")).toContainText(
      "运行 completed",
      { timeout: 45_000 },
    );

    await page.getByRole("button", { name: "完整报告", exact: true }).click();
    const issueCards = page.locator(".issues article");
    await expect(issueCards).toHaveCount(1);
    await expect(issueCards.first()).toContainText("事实冲突");
    await expect(issueCards.first()).toContainText("world.docx:2");
    await expect(issueCards.first()).toContainText("林澈的发色是银色。");
    await expect(issueCards.first()).toContainText("chapter.docx:2");
    await expect(issueCards.first()).toContainText("林澈的发色是黑色。");
    const issueTexts = await issueCards.allTextContents();
    expect(new Set(issueTexts).size).toBe(issueTexts.length);

    await page.getByRole("button", { name: "运行审计" }).first().click();
    await page
      .getByText(/查看系统实际抽取记录/)
      .click();
    const recordCards = page.locator(".recordCard");
    await expect(recordCards).toHaveCount(2);
    const recordKeys = await recordCards.evaluateAll((cards) =>
      cards.map((card) => {
        const summary = card.querySelector("p")?.textContent?.trim() || "";
        const source = card.querySelector("small")?.textContent?.trim() || "";
        return `${summary}|${source}`;
      }),
    );
    expect(new Set(recordKeys).size).toBe(recordKeys.length);
  } finally {
    if (workerPaused) compose("unpause", "worker");
  }

  expect([...browserApiOrigins]).toEqual(["http://127.0.0.1:8080"]);
});
