import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { execSync } from "node:child_process";

const DIST_AA_STATS_PATH = resolve("dist/clients/aaStats.js");

if (!existsSync(DIST_AA_STATS_PATH)) {
  execSync("npm run build", { stdio: "inherit" });
}

const { getAAStats } = await import(`file://${DIST_AA_STATS_PATH}`);
await getAAStats();
