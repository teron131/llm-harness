import * as agents from "./agents/index.js";
import * as clients from "./clients/index.js";
import * as tools from "./tools/index.js";
import * as utils from "./utils/index.js";
import { getAAStats } from "./clients/aaStats.js";

const AUTO_REFRESH_AA_STATS = process.env.AA_AUTO_REFRESH !== "0";

if (AUTO_REFRESH_AA_STATS) {
  void getAAStats().catch((error) => {
    if (process.env.NODE_ENV !== "test") {
      console.warn("AA stats auto-refresh failed:", error);
    }
  });
}

export { agents, clients, tools, utils };

export * from "./agents/index.js";
export * from "./clients/index.js";
export * from "./tools/index.js";
export * from "./utils/index.js";
