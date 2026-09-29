import { timestampFromDate } from "@bufbuild/protobuf/wkt";
import { Code } from "@connectrpc/connect";
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";
import { after, afterEach, before, beforeEach, describe, test } from "node:test";
import { fileURLToPath } from "node:url";
import { PassThrough, Readable } from "node:stream";
import { promisify } from "node:util";

import type Anthropic from "@anthropic-ai/sdk";
import { create, Origin } from "jennah-sdk-ts";
import {
  CandidateKind,
  CommitMemoryResponseSchema,
  ConversationTurnSchema,
  EntityClassSchema,
  FormedCandidateSchema,
  FormMemoryResponseSchema,
  GetMemoryVocabularyResponseSchema,
  InspectMemoryResponseSchema,
  MemoryDecision,
  QueryMemoryResponseSchema,
  RelationTypeSchema,
  TurnRole,
  type CommitMemoryRequest,
  type FormedCandidate,
  type FormMemoryRequest,
  type InspectMemoryRequest,
  type QueryMemoryRequest,
} from "jennah-sdk-ts/gen/jennah/agent/v1/memory_pb";
import type { CreateAgentRequest } from "jennah-sdk-ts/gen/jennah/agent/v1/agent_pb";

import { commitRequest, nodeId, normRel, seedRequest, type Fact } from "../src/authored.js";
import { AnthropicBrain, GeminiBrain, selectProvider, type GeminiModels } from "../src/brain.js";
import * as jennah from "../src/jennah.js";
import { connect, main, parseFlags } from "../src/main.js";
import { commitLines, formationLines, type Line } from "../src/receipt.js";
import { API_KEY } from "./fake.js";
import { chat, emptyMachine, Events, FakeBrain, noInput, tmpDir, world, type Machine, type World } from "./helpers.js";

const CLI = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "src", "cli.js");
const DAY = 86_400_000;

let w: World;
beforeEach(async () => {
  w = await world();
});
afterEach(() => w.close());

const texts = (lines: Line[]): string[] => lines.map(([, t]) => t);

// ---- 2.1 flags: secrets stay out of --help ----

test("--help does not leak a key", async () => {
  const env = {
    ...process.env,
    JENNAH_API_KEY: "jennah_sk_" + "Q7xS3cr3tV4lu3Zz9",
    ANTHROPIC_API_KEY: "sk-ant-An0th3rS3cr3t",
  };
  const { stdout } = await promisify(execFile)(process.execPath, [CLI, "--help"], { env });
  assert.ok(stdout.includes("--jennah-api-key") && stdout.includes("--authored"));
  for (const part of ["Q7xS3cr3tV4lu3Zz9", "S3cr3t", "An0th3r"]) assert.ok(!stdout.includes(part), part);
});

test("an unknown flag is refused", async () => {
  const ev = new Events();
  assert.equal(await main(["--nope"], { out: ev.out }), 2);
  assert.match(ev.text(), /--nope/);
});

// ---- 2.2 credentials ----

describe("credentials", () => {
  let m: Machine;
  before(() => {
    m = emptyMachine();
  });
  after(() => m.restore());

  test("no credential exits before any call", async () => {
    const ev = new Events();
    const code = await main(["--endpoint", w.fake.endpoint, "--insecure", "--provider", "gemini"], {
      out: ev.out,
      brain: () => new FakeBrain(),
    });
    assert.equal(code, 1);
    const err = ev.text();
    assert.ok(err.includes("--jennah-api-key") && err.includes("JENNAH_API_KEY") && err.includes("jnh login"), err);
    assert.deepEqual(w.fake.calls, []);
  });

  test("a signed-in session is used", async () => {
    fs.mkdirSync(path.dirname(m.sessionPath), { recursive: true, mode: 0o700 });
    fs.writeFileSync(
      m.sessionPath,
      JSON.stringify({ endpoint: "https://jennah.alphaus.cloud", access_token: API_KEY, token_type: "Bearer", expires_at: 0 }),
      { mode: 0o600 },
    );
    const client = connect(parseFlags(["--endpoint", w.fake.endpoint, "--insecure"], {}));
    try {
      assert.equal(client.credential.origin, Origin.File);
      w.fake.agents.add("demo.s");
      await jennah.requireAgent(client, "demo.s");
    } finally {
      client.close();
    }
  });
});

// ---- 2.3 workspace resolution never guesses ----

test("first run creates, later runs reuse", async () => {
  const state = path.join(tmpDir(), "state.json");
  const first = await jennah.resolveWorkspace(w.client, { agent: "", statePath: state, region: "asia-northeast1" });
  assert.ok(first.agentId.startsWith("demo.") && first.where.includes("created") && first.where.includes("asia-northeast1"));
  const created = w.fake.of("CreateAgent");
  assert.equal(created.length, 1);
  assert.equal((created[0]!.request as CreateAgentRequest).region, "asia-northeast1");
  assert.deepEqual(JSON.parse(fs.readFileSync(state, "utf8")), { agent_id: first.agentId });

  const second = await jennah.resolveWorkspace(w.client, { agent: "", statePath: state, region: "us-central1" });
  assert.ok(second.agentId === first.agentId && second.where.includes("reusing"));
  assert.equal(w.fake.of("CreateAgent").length, 1);
});

test("a mistyped agent id fails naming both causes", async () => {
  const state = path.join(tmpDir(), "s.json");
  await assert.rejects(
    jennah.resolveWorkspace(w.client, { agent: "demo.typo", statePath: state, region: "" }),
    (e: Error) => {
      assert.ok(e instanceof jennah.StartupError);
      assert.ok(e.message.includes("demo.typo") && e.message.includes("not there") && e.message.includes("cannot reach"));
      return true;
    },
  );
  assert.deepEqual(w.fake.of("CreateAgent"), []);
  assert.ok(!fs.existsSync(state));
});

test("a mistyped agent id stops startup before the first prompt", async () => {
  const ev = new Events();
  const code = await main(
    ["--endpoint", w.fake.endpoint, "--insecure", "--jennah-api-key", API_KEY, "--agent", "demo.typo"],
    { out: ev.out, brain: () => new FakeBrain(), ...noInput() },
  );
  assert.equal(code, 1);
  assert.ok(!ev.text().includes("Ctrl-D"));
});

test("--agent leaves the state file alone", async () => {
  const state = path.join(tmpDir(), "s.json");
  fs.writeFileSync(state, '{"agent_id": "demo.mine"}');
  const before = fs.readFileSync(state);
  w.fake.agents.add("demo.operator");
  const got = await jennah.resolveWorkspace(w.client, { agent: "demo.operator", statePath: state, region: "x" });
  assert.ok(got.agentId === "demo.operator" && got.where.includes("untouched"));
  assert.deepEqual(fs.readFileSync(state), before);
  assert.deepEqual(w.fake.of("CreateAgent"), []);
});

// ---- 2.4 recall ----

test("recall pages and filters retired edges", async () => {
  const now = Date.now();
  w.fake.inspectPages = [
    create(InspectMemoryResponseSchema, {
      graph: {
        nodes: [{ nodeId: "n1", label: "Chew" }],
        edges: [
          {
            edgeId: "e1",
            sourceNodeId: "n1",
            targetNodeId: "n2",
            relationshipType: "LIVES_IN",
            invalidAt: timestampFromDate(new Date(now - DAY)),
          },
          { edgeId: "e2", sourceNodeId: "n1", targetNodeId: "n3", relationshipType: "LIVES_IN" },
        ],
      },
      nextNodeToken: "more-nodes",
    }),
    create(InspectMemoryResponseSchema, {
      graph: {
        nodes: [
          { nodeId: "n2", label: "Osaka" },
          { nodeId: "n3", label: "Tokyo" },
        ],
        edges: [
          {
            edgeId: "e3",
            sourceNodeId: "n1",
            targetNodeId: "n2",
            relationshipType: "VISITED",
            invalidAt: timestampFromDate(new Date(now + DAY)),
          },
        ],
      },
    }),
  ];
  const { facts, retired } = await jennah.recallFacts(w.client, "demo.a");
  assert.deepEqual(facts, ["Chew lives in Tokyo", "Chew visited Osaka"]);
  assert.equal(retired, 1);
  const reqs = w.fake.of("InspectMemory").map((c) => c.request as InspectMemoryRequest);
  assert.equal(reqs.length, 2);
  assert.ok(reqs[0]!.graph?.nodeLimit === 200 && reqs[0]!.graph?.edgeLimit === 200);
  assert.ok(reqs[1]!.graph?.nodePageToken === "more-nodes" && reqs[1]!.graph?.edgePageToken === "");
});

test("recall stops at ten pages", async () => {
  w.fake.inspectPages = [create(InspectMemoryResponseSchema, { nextEdgeToken: "again" })];
  await jennah.recallFacts(w.client, "demo.a");
  assert.equal(w.fake.of("InspectMemory").length, 10);
});

test("the retired count is shown and the edge is kept out of the prompt", async () => {
  w.fake.inspectPages = [
    create(InspectMemoryResponseSchema, {
      graph: {
        nodes: [
          { nodeId: "a", label: "Chew" },
          { nodeId: "b", label: "Osaka" },
        ],
        edges: [
          {
            edgeId: "e1",
            sourceNodeId: "a",
            targetNodeId: "b",
            relationshipType: "LIVES_IN",
            invalidAt: timestampFromDate(new Date(Date.now() - 5000)),
          },
        ],
      },
    }),
  ];
  const { c, ev, brain } = chat(w);
  await c.turn("hi");
  assert.ok(ev.text().includes("[recalled 0 fact(s) (+1 retired, not shown), 0 past snippet(s)]"), ev.text());
  assert.ok(!brain.systems[0]!.includes("Osaka"));
});

test("provenance is verbose-only and never in the prompt", async () => {
  w.fake.query = create(QueryMemoryResponseSchema, {
    semantic: {
      matches: [
        {
          chunkId: "c1",
          rawContent: "User likes   hiking",
          metadata: { "jennah.source_step": "step_formed_42", "jennah.source_turns": "0-1" },
        },
      ],
    },
  });
  const v = chat(w, { verbose: true });
  await v.c.turn("what do I like?");
  const sys = v.brain.systems[0]!;
  assert.ok(sys.includes("- User likes hiking"));
  assert.ok(!sys.includes("step_formed_42") && !sys.includes("formed by"));
  assert.ok(v.ev.text().includes("~ User likes hiking  [formed by step_formed_42, turn(s) 0-1]"), v.ev.text());

  const q = chat(w);
  await q.c.turn("again");
  assert.ok(!q.ev.text().includes("step_formed_42"));
});

test("the semantic query carries the message and limit 6", async () => {
  await jennah.recallSemantic(w.client, "demo.a", "hello");
  const req = w.fake.of("QueryMemory")[0]!.request as QueryMemoryRequest;
  assert.ok(req.semantic?.queryText === "hello" && req.semantic?.limit === 6);
});

// ---- 2.5 vocabulary banner ----

test("vocabulary counts", async () => {
  w.fake.vocabulary = create(GetMemoryVocabularyResponseSchema, {
    resolved: {
      entityClasses: [create(EntityClassSchema), create(EntityClassSchema)],
      relationTypes: [create(RelationTypeSchema)],
    },
  });
  assert.equal(await jennah.vocabularySummary(w.client, "demo.a"), "2 entity class(es), 1 relation type(s) in effect");
  assert.equal((w.fake.of("GetMemoryVocabulary")[0]!.request as { scopeId: string }).scopeId, "demo.a");
});

test("vocabulary: none declared", async () => {
  const got = await jennah.vocabularySummary(w.client, "demo.a");
  assert.ok(got.startsWith("none declared") && got.includes("jnh vocabulary declare --scope demo.a"));
});

test("vocabulary: permission denied reads as a missing permission", async () => {
  w.fake.vocabulary = Code.PermissionDenied;
  const got = await jennah.vocabularySummary(w.client, "demo.a");
  assert.ok(got.includes("agent.vocabulary:read") && got.includes("not readable"));
});

test("startup survives a denied vocabulary", async () => {
  w.fake.vocabulary = Code.PermissionDenied;
  const ev = new Events();
  const code = await main(
    [
      "--endpoint",
      w.fake.endpoint,
      "--insecure",
      "--jennah-api-key",
      API_KEY,
      "--state",
      path.join(tmpDir(), "s.json"),
    ],
    { out: ev.out, brain: () => new FakeBrain(), ...noInput() },
  );
  assert.equal(code, 0);
  assert.ok(ev.text().includes("vocabulary: not readable") && ev.text().includes("bye."), ev.text());
});

// ---- 3.1 brain selection, and no tools by default ----

test("auto provider selection", async () => {
  assert.equal(selectProvider("auto", "sk-ant-x", {}), "anthropic");
  assert.equal(selectProvider("auto", "sk-ant-x", { GEMINI_API_KEY: "g" }), "anthropic");
  assert.equal(selectProvider("auto", "", { GEMINI_API_KEY: "g" }), "gemini");
  assert.equal(selectProvider("auto", "", { GOOGLE_API_KEY: "g" }), "gemini");
  assert.equal(selectProvider("auto", "", { GOOGLE_CLOUD_PROJECT: "jennah-hq" }), "gemini");
  assert.equal(selectProvider("auto", "", { GOOGLE_GENAI_USE_VERTEXAI: "true" }), "gemini");
  assert.throws(() => selectProvider("auto", "", {}), /no chat credentials/);
  assert.throws(() => selectProvider("openai", "", {}), /unknown --provider/);
});

type Req = Record<string, unknown> & { messages: unknown[] };

function anthropicStub(responses: unknown[]): { client: Anthropic; requests: Req[] } {
  const requests: Req[] = [];
  const client = {
    messages: {
      create: async (p: Req) => {
        // history is mutated after the call, so keep the messages as sent
        requests.push({ ...p, messages: [...p.messages] });
        return responses.shift();
      },
    },
  } as unknown as Anthropic;
  return { client, requests };
}

const textResp = (text: string) => ({ content: [{ type: "text", text }], stop_reason: "end_turn" });

test("anthropic: the default arm sends no tools", async () => {
  const stub = anthropicStub([textResp("hello")]);
  const b = new AnthropicBrain("", false, stub.client);
  assert.deepEqual(await b.chat("sys", "hi"), { reply: "hello", facts: [] });
  assert.ok(!("tools" in stub.requests[0]!));
  assert.equal(stub.requests[0]!.system, "sys");
});

test("anthropic: the authored arm collects facts", async () => {
  const stub = anthropicStub([
    { content: [{ type: "tool_use", id: "t1", input: { relationship: "lives in", object: "Tokyo" } }], stop_reason: "tool_use" },
    textResp("noted!"),
  ]);
  const b = new AnthropicBrain("", true, stub.client);
  const { reply, facts } = await b.chat("sys", "I live in Tokyo");
  assert.equal(reply, "noted!");
  assert.deepEqual(facts, [{ subj: "", rel: "lives in", obj: "Tokyo" }]);
  assert.equal((stub.requests[0]!.tools as { name: string }[])[0]!.name, "remember_fact");
  const last = stub.requests[1]!.messages.at(-1) as { content: { tool_use_id: string }[] };
  assert.equal(last.content[0]!.tool_use_id, "t1");
});

function geminiStub(text: string): { models: GeminiModels; configs: Record<string, unknown>[] } {
  const configs: Record<string, unknown>[] = [];
  const models = {
    generateContent: async (p: { config: Record<string, unknown> }) => {
      configs.push(p.config);
      return { candidates: [{ content: { role: "model", parts: [{ text }] } }] };
    },
  } as unknown as GeminiModels;
  return { models, configs };
}

test("gemini: the default arm sends no tools", async () => {
  const stub = geminiStub("hi there");
  const b = new GeminiBrain(false, { GEMINI_API_KEY: "k" }, stub.models);
  assert.deepEqual(await b.chat("sys", "hi"), { reply: "hi there", facts: [] });
  assert.ok(!stub.configs[0]!.tools);
  assert.ok(b.label.endsWith("(ai-studio)"));
});

test("gemini: the authored arm offers remember_fact", async () => {
  const stub = geminiStub("ok");
  await new GeminiBrain(true, { GEMINI_API_KEY: "k" }, stub.models).chat("sys", "hi");
  const tools = stub.configs[0]!.tools as { functionDeclarations: { name: string; parameters: { properties: object; required: string[] } }[] }[];
  const decl = tools[0]!.functionDeclarations[0]!;
  assert.equal(decl.name, "remember_fact");
  assert.deepEqual(Object.keys(decl.parameters.properties).sort(), ["object", "relationship", "subject"]);
  assert.deepEqual(decl.parameters.required, ["relationship", "object"]);
});

test("gemini: the vertex label", async () => {
  const b = new GeminiBrain(false, { GOOGLE_CLOUD_PROJECT: "jennah-hq" }, geminiStub("x").models);
  assert.equal(b.label, "gemini/gemini-3.8-flash (vertex:jennah-hq/global)");
});

test("the default prompt has no store instruction", async () => {
  const { c, brain } = chat(w);
  await c.turn("hi");
  assert.ok(!brain.systems[0]!.includes("remember_fact") && !brain.systems[0]!.toLowerCase().includes("store"));
});

// ---- 3.2 the turn loop ----

test("the window is the last six turns", async () => {
  const { c, brain } = chat(w);
  for (let i = 0; i < 5; i++) {
    brain.reply = `reply ${i}`;
    await c.turn(`message ${i}`);
  }
  const forms = w.fake.of("FormMemory").map((x) => x.request as FormMemoryRequest);
  // 5 exchanges is 10 turns; the fifth formation submits only the last 6.
  const last = forms.at(-1)!;
  assert.deepEqual(
    last.turns.map((t) => t.content),
    ["message 2", "reply 2", "message 3", "reply 3", "message 4", "reply 4"],
  );
  assert.ok(last.turns[0]!.role === TurnRole.USER && last.turns[1]!.role === TurnRole.ASSISTANT);
  assert.equal(forms[0]!.turns.length, 2);
});

test("the same text on different turns gets different keys", async () => {
  const { c } = chat(w);
  await c.turn("thanks");
  await c.turn("thanks");
  const keys = w.fake.of("FormMemory").map((x) => (x.request as FormMemoryRequest).formationKey);
  assert.equal(new Set(keys).size, 2);
  assert.deepEqual(keys, [`frm_${c.sessionId}_1`, `frm_${c.sessionId}_2`]);
  assert.equal(w.fake.extractions, 2);
});

test("a resent formation replays", async () => {
  const turns = [create(ConversationTurnSchema, { role: TurnRole.USER, content: "x" })];
  w.fake.form = () => create(FormMemoryResponseSchema, { vectorRows: 1n });
  const a = await jennah.form(w.client, "demo.a", turns, "frm_s_1");
  const b = await jennah.form(w.client, "demo.a", turns, "frm_s_1");
  assert.equal(a.vectorRows, b.vectorRows);
  assert.equal(w.fake.extractions, 1);
});

test("deadlines on the wire", async () => {
  const { c } = chat(w);
  await c.turn("hi");
  const form = w.fake.of("FormMemory")[0]!.timeoutMs!;
  assert.ok(form > 290_000 && form <= 300_000, String(form));
  for (const method of ["QueryMemory", "InspectMemory"]) {
    const d = w.fake.of(method)[0]!.timeoutMs!;
    assert.ok(d > 0 && d <= 60_000, `${method} ${d}`);
  }
});

test("the commit deadline is ordinary", async () => {
  const { c } = chat(w, { authored: true });
  await c.turn("hi");
  const d = w.fake.of("CommitMemory")[0]!.timeoutMs!;
  assert.ok(d > 0 && d <= 60_000, String(d));
});

test("a slow formation is waited for", async () => {
  // Scaled down from the spec's 90 s: what matters is that a formation slower
  // than any ordinary call's budget is not abandoned.
  w.fake.delay.FormMemory = 1500;
  w.fake.form = () => create(FormMemoryResponseSchema, { candidates: [cand(MemoryDecision.NEW, "x")] });
  const { c, ev } = chat(w);
  await c.turn("hi");
  assert.ok(ev.text().includes("[formed: 1 new]"), ev.text());
  assert.ok(!ev.lines.some(([s]) => s === "err"));
});

test("the reply is printed before the formation", async () => {
  const { c, ev } = chat(w, { brain: new FakeBrain("the answer") });
  w.fake.form = (req) => {
    ev.timeline.push(["form", req.formationKey]);
    return create(FormMemoryResponseSchema);
  };
  await c.turn("question");
  const kinds = ev.timeline.filter(([k, t]) => k === "form" || t.includes("memo> the answer")).map(([k]) => k);
  assert.deepEqual(kinds, ["out", "form"]);
});

test("a formation failure is a warning", async () => {
  w.fake.fail.FormMemory = Code.Unavailable;
  const { c, ev } = chat(w);
  await c.turn("hi");
  assert.ok(ev.lines.some(([s, t]) => s === "err" && t.includes("could not form this turn's memory")));
});

test("the session loop answers each line and stops at /exit", async () => {
  const ev = new Events();
  const output = new PassThrough();
  output.resume();
  const code = await main(
    ["--endpoint", w.fake.endpoint, "--insecure", "--jennah-api-key", API_KEY, "--state", path.join(tmpDir(), "s.json")],
    {
      out: ev.out,
      brain: () => new FakeBrain("hello back"),
      input: Readable.from(["hi there\n", "\n", "/exit\n", "never read\n"]),
      output,
    },
  );
  assert.equal(code, 0);
  assert.equal(ev.lines.filter(([, t]) => t.includes("memo> hello back")).length, 1);
  assert.equal(w.fake.of("FormMemory").length, 1);
  assert.ok(ev.text().endsWith("bye. Your memory is saved in Jennah."));
});

// ---- 3.3 receipt rendering ----

function cand(decision: MemoryDecision, text = "", extra: { matchedId?: string; rejectionReason?: string } = {}): FormedCandidate {
  return create(FormedCandidateSchema, { decision, text, kind: CandidateKind.FACT, ...extra });
}

test("receipt: counts", async () => {
  const r = create(FormMemoryResponseSchema, {
    candidates: [cand(MemoryDecision.NEW), cand(MemoryDecision.NEW), cand(MemoryDecision.REVISED), cand(MemoryDecision.REJECTED)],
  });
  assert.deepEqual(texts(formationLines(r, false)), ["[formed: 2 new, 1 revised, 1 rejected]"]);
});

test("receipt: nothing worth remembering", async () => {
  assert.deepEqual(texts(formationLines(create(FormMemoryResponseSchema), false)), [
    "[formed: nothing worth remembering in that exchange]",
  ]);
});

test("receipt: a supersession is shown without verbose", async () => {
  const r = create(FormMemoryResponseSchema, { candidates: [cand(MemoryDecision.REVISED)], edgeSupersessions: 1n });
  assert.deepEqual(formationLines(r, false)[1], [
    "cyan",
    "[memory] 1 earlier assertion(s) retired by a correction in this turn " +
      "(superseded, not overwritten: the previous value stays readable as history)",
  ]);
});

test("receipt: dropped, summarized and redacted", async () => {
  const r = create(FormMemoryResponseSchema, {
    candidates: [cand(MemoryDecision.NEW)],
    candidatesDropped: 3,
    candidateCap: 20,
    summarizedStructures: [{ turnIndex: 2, description: "a table of prices", summarizedCount: 14 }],
    redactions: [{ turnIndex: 0, maskedCount: 2 }],
  });
  assert.deepEqual(texts(formationLines(r, false)).slice(1), [
    "[memory] 3 candidate(s) past the per-formation cap of 20 were dropped, " +
      "so not everything in that exchange was considered.",
    "[memory] turn 2: a table of prices (14 item(s) summarized rather than stored individually)",
    "[memory] turn 0: 2 value(s) masked before the extraction model saw them",
  ]);
});

test("receipt: verbose notes", async () => {
  const r = create(FormMemoryResponseSchema, {
    candidates: [
      create(FormedCandidateSchema, {
        decision: MemoryDecision.REVISED,
        kind: CandidateKind.RELATIONSHIP,
        sourceEntity: "Chew",
        relationshipType: "LIVES_IN",
        targetEntity: "Tokyo",
        matchedId: "e_old",
      }),
      cand(MemoryDecision.KNOWN, "likes  hiking", { matchedId: "c_1" }),
      cand(MemoryDecision.REJECTED, "used to live in Osaka", { rejectionReason: "recounts history" }),
      cand(MemoryDecision.NEW, "has a cat"),
    ],
    vectorRows: 1n,
    graphEdgeRows: 0n,
    edgeSupersessions: 1n,
    executionLogRows: 1n,
  });
  const got = texts(formationLines(r, true));
  assert.deepEqual(got.slice(0, 4), [
    "revised  Chew lives in Tokyo  (retired e_old)",
    "known    likes hiking  (matches c_1)",
    "rejected used to live in Osaka  (recounts history)",
    "new      has a cat",
  ]);
  assert.equal(got[4], "formed: log=1 vec=1(+0 superseded) nodes=0 edges=0(+1 superseded) @ nothing committed");
  assert.ok(got[5]!.startsWith("[memory] 1 earlier assertion(s) retired"));
});

// ---- 4.1 authored arm: label convergence ----

const fact = (subj: string, rel: string, obj: string): Fact => ({ subj, rel, obj });

test("two phrasings converge on one edge", async () => {
  const a = commitRequest("demo.a", "u", "r", [fact("Chew", "is CTO of", "NightBlue")]).req;
  const b = commitRequest("demo.a", "u", "r", [fact("NightBlue", "has CTO", "Chew")]).req;
  const ea = a.graph!.edges[0]!;
  const eb = b.graph!.edges[0]!;
  assert.equal(ea.edgeId, eb.edgeId);
  assert.deepEqual(
    [ea.sourceNodeId, ea.relationshipType, ea.targetNodeId],
    [nodeId("NightBlue"), "HAS_CTO", nodeId("Chew")],
  );
});

test("ids match the other memchat clients", async () => {
  // The same workspace can be written by any memchat, so the content-hashed ids
  // must agree byte for byte. These are the Python client's values.
  assert.equal(nodeId("NightBlue"), "n_351f03acd9af");
});

test("user references fold onto the anchor", async () => {
  for (const ref of ["", "me", "I", "the user", " User ", "myself"]) assert.equal(nodeId(ref), "user", ref);
  assert.equal(nodeId("Chew"), nodeId("  chew "));
  assert.equal(normRel("is named"), "IS_NAMED");
  assert.equal(normRel("  "), "RELATED_TO");
  const { req, stored } = commitRequest("demo.a", "u", "r", [fact("", "lives in", "Tokyo")]);
  assert.deepEqual(
    req.graph!.nodes.map((n) => n.nodeId),
    [nodeId("Tokyo")],
  );
  assert.deepEqual(stored, ["user lives in Tokyo"]);
});

test("the seed is the user anchor", async () => {
  assert.deepEqual(
    seedRequest("demo.a").graph!.nodes.map((n) => [n.nodeId, n.label]),
    [["user", "User"]],
  );
});

test("an authored start seeds the anchor and skips the vocabulary", async () => {
  const ev = new Events();
  const code = await main(
    [
      "--endpoint",
      w.fake.endpoint,
      "--insecure",
      "--jennah-api-key",
      API_KEY,
      "--authored",
      "--state",
      path.join(tmpDir(), "s.json"),
    ],
    { out: ev.out, brain: () => new FakeBrain(), ...noInput() },
  );
  assert.equal(code, 0);
  const seeds = w.fake.of("CommitMemory");
  assert.equal(seeds.length, 1);
  assert.equal((seeds[0]!.request as CommitMemoryRequest).graph!.nodes[0]!.nodeId, "user");
  assert.deepEqual(w.fake.of("GetMemoryVocabulary"), []);
});

// ---- 4.2 the turn commit ----

test("one commit carries all three sections, and truncation is disclosed", async () => {
  w.fake.commit = create(CommitMemoryResponseSchema, { truncatedChunkIds: ["chunk_big"] });
  const facts = [fact("", "lives in", "Tokyo"), fact("", "lives in", "Tokyo"), fact("Chew", "is CTO of", "NightBlue")];
  const { c, ev } = chat(w, { authored: true, brain: new FakeBrain("got it", facts) });
  await c.turn("I live in Tokyo, and Chew is CTO of NightBlue");
  const commits = w.fake.of("CommitMemory");
  assert.equal(commits.length, 1);
  const req = commits[0]!.request as CommitMemoryRequest;
  assert.equal(req.log!.toolUsed, "memchat");
  assert.equal(req.vectors[0]!.rawContent, "User: I live in Tokyo, and Chew is CTO of NightBlue\nAssistant: got it");
  assert.equal(req.graph!.edges.length, 2); // the repeated fact is deduplicated within the commit
  assert.deepEqual(new Set(req.graph!.nodes.map((n) => n.label)), new Set(["Tokyo", "Chew", "NightBlue"]));
  assert.deepEqual(w.fake.of("FormMemory"), []);
  assert.ok(
    ev.lines.some(
      ([s, t]) =>
        s === "yellow" &&
        t === "[memory] that message was too long to embed in full (chunk_big). It is stored, but recall may miss the end of it.",
    ),
  );
});

test("the commit receipt is quiet without truncation", async () => {
  assert.deepEqual(commitLines(create(CommitMemoryResponseSchema, { vectorRows: 1n }), false), []);
});
