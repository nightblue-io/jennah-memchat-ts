import assert from "node:assert/strict";
import { test } from "node:test";

import { world } from "./helpers.js";

test("the fake is reached through the SDK's own client", async () => {
  const w = await world();
  try {
    w.fake.agents.add("demo.smoke");
    const resp = await w.client.agents.getAgent({ agentInstanceId: "demo.smoke" }, { timeoutMs: 5000 });
    assert.equal(resp.agent?.agentInstanceId, "demo.smoke");
    const [call] = w.fake.of("GetAgent");
    assert.ok(call && call.timeoutMs !== undefined && call.timeoutMs <= 5000);
  } finally {
    await w.close();
  }
});
