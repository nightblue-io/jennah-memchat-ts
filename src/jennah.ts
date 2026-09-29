// Every Jennah call memchat makes, through the SDK's Client.
//
// Nothing here builds a request to the platform by hand or opens a connection
// of its own: each call is a method on `client.agents` or `client.memory`, so
// credential resolution, session renewal and retry classification are the
// SDK's.

import * as fs from "node:fs";

import { Code, ConnectError, code, type Client } from "jennah-sdk-ts";
import type {
  CommitMemoryRequest,
  CommitMemoryResponse,
  ConversationTurn,
  FormMemoryResponse,
  GraphEdge,
} from "jennah-sdk-ts/gen/jennah/agent/v1/memory_pb";

import { randId, tripleText } from "./authored.js";

// CALL_TIMEOUT_MS is the deadline for the millisecond-class calls (query,
// inspect, commit, workspace checks). FORM_TIMEOUT_MS is for memory:form alone,
// and it is five times longer on purpose.
//
// A formation is slow by contract, not by accident: it runs two bounded
// generative calls plus a retrieval before it writes, and the platform checks at
// startup that the whole thing fits the 300 s its load balancer allows a
// request. A 60 s client deadline would abandon formations the server goes on to
// finish and commit, and report memory that WAS written as a failure. The
// formation key makes that recoverable, but recovery should not be the ordinary
// path.
export const CALL_TIMEOUT_MS = 60_000;
export const FORM_TIMEOUT_MS = 300_000;

// Every workspace this demo creates is named under "demo.". '.' is the
// platform's agent-selector separator and selector matching is
// segment-anchored, so one role selector "demo.*" reaches every id minted here
// and nothing else. That lets a demo run be scoped to a throwaway role instead
// of needing blanket agent access.
export const DEMO_PREFIX = "demo.";

// The graph read-out's page size, and a bound on the walk so a runaway
// workspace cannot stall a chat turn.
export const RECALL_PAGE_LIMIT = 200;
export const MAX_RECALL_PAGES = 10;

export const SEMANTIC_LIMIT = 6;

/** A condition that stops the demo before the first prompt. */
export class StartupError extends Error {}

/** Per-call options: the deadline, and the signal that Ctrl-C aborts. */
export interface CallOpts {
  signal?: AbortSignal;
}

/** An error as one line, with the platform's status name when it has one. */
export function describe(err: unknown): string {
  if (err instanceof ConnectError) return `${Code[err.code]}: ${err.rawMessage}`;
  return err instanceof Error ? err.message : String(err);
}

// ---- workspace resolution ----

/** The workspace id in the state file, or "" when there is none yet. */
export function loadState(path: string): string {
  let raw: string;
  try {
    raw = fs.readFileSync(path, "utf8");
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return "";
    throw e;
  }
  const v = (JSON.parse(raw) as { agent_id?: unknown }).agent_id;
  return typeof v === "string" ? v : "";
}

/**
 * Persist the workspace id, and only that. Nothing else needs tracking: graph
 * writes are idempotent upserts on caller-supplied ids, so re-asserting a fact
 * across sessions converges instead of duplicating.
 */
export function saveState(path: string, agentId: string): void {
  fs.writeFileSync(path, JSON.stringify({ agent_id: agentId }, null, 2), { mode: 0o600 });
}

/**
 * Check that a workspace named with --agent is really there, so a mistyped id
 * fails at startup naming itself rather than three lines into a conversation.
 *
 * A not-found is indistinguishable from a workspace this credential cannot
 * reach, because the platform collapses the two on purpose (a refusal that
 * confirmed the id exists would be a disclosure). So the message names both.
 */
export async function requireAgent(client: Client, agentId: string): Promise<void> {
  try {
    await client.agents.getAgent({ agentInstanceId: agentId }, { timeoutMs: CALL_TIMEOUT_MS });
  } catch (e) {
    if (code(e) === Code.NotFound) {
      throw new StartupError(
        `agent workspace '${agentId}' is not there, or this credential cannot reach it ` +
          `(the platform answers both the same way). Create it with 'jnh agents create ${agentId}', ` +
          "or drop --agent to let this demo mint its own",
      );
    }
    throw new StartupError(`check agent workspace '${agentId}': ${describe(e)}`);
  }
}

/**
 * Provision a workspace. region is applied only here, because an agent is
 * pinned to one home region for its lifetime; "" means the platform default.
 */
export async function createAgent(client: Client, region: string): Promise<string> {
  const wanted = DEMO_PREFIX + randId("memchat");
  const resp = await client.agents.createAgent(
    { agentInstanceId: wanted, agentName: "memchat-demo", region },
    { timeoutMs: CALL_TIMEOUT_MS },
  );
  return resp.agent?.agentInstanceId || wanted;
}

/**
 * The workspace this run uses, and the startup line that says where it came
 * from.
 *
 * --agent names a workspace someone else provisioned, which is the ordinary way
 * to run against one with a vocabulary declared on it. It never creates and
 * never touches the state file. Creating would let a mistyped id silently mint
 * a second, empty workspace, and the demo would then report remembering
 * nothing, which reads as a platform fault. Writing the id to the state file
 * would leave later flagless runs pointed at the operator's workspace.
 */
export async function resolveWorkspace(
  client: Client,
  opts: { agent: string; statePath: string; region: string },
): Promise<{ agentId: string; where: string }> {
  const agent = opts.agent.trim();
  if (agent) {
    await requireAgent(client, agent);
    return { agentId: agent, where: `using agent workspace ${agent} (--agent; the state file is untouched)` };
  }
  let agentId = loadState(opts.statePath);
  if (agentId) return { agentId, where: `reusing agent workspace ${agentId} (memory carries over)` };
  agentId = await createAgent(client, opts.region);
  saveState(opts.statePath, agentId);
  const where = opts.region ? `region ${opts.region}` : "platform default region";
  return { agentId, where: `created agent workspace ${agentId} (${where})` };
}

// ---- recall ----

/**
 * One recalled chunk: the text for the prompt, and separately where it came
 * from. Provenance is for the person watching --verbose; splicing it into the
 * prompt would put ids in front of the model that it has no use for.
 */
export interface Snippet {
  text: string;
  prov: string;
}

export interface Recall {
  facts: string[];
  snippets: Snippet[];
  retired: number;
}

/**
 * Where a recalled chunk came from, when a formation wrote it.
 *
 * Formation stamps what it writes under reserved jennah.* keys a caller cannot
 * set itself. An authored chunk carries none, so this is "" for it.
 */
export function provenance(metadata: Record<string, string>): string {
  const step = metadata["jennah.source_step"] ?? "";
  if (!step) return "";
  const turns = metadata["jennah.source_turns"] ?? "";
  return turns ? `  [formed by ${step}, turn(s) ${turns}]` : `  [formed by ${step}]`;
}

export async function recallSemantic(client: Client, agentId: string, query: string, o: CallOpts = {}): Promise<Snippet[]> {
  const resp = await client.memory.queryMemory(
    { agentInstanceId: agentId, semantic: { queryText: query, limit: SEMANTIC_LIMIT } },
    { timeoutMs: CALL_TIMEOUT_MS, signal: o.signal },
  );
  const out: Snippet[] = [];
  for (const m of resp.semantic?.matches ?? []) {
    const content = m.rawContent.trim();
    if (content) out.push({ text: content.split(/\s+/).join(" "), prov: provenance(m.metadata) });
  }
  return out;
}

/**
 * Whether an edge's valid-time window has closed as of now. An unset invalidAt
 * means still current, and a future one is still live.
 */
export function isRetired(edge: GraphEdge, now: Date): boolean {
  if (!edge.invalidAt) return false;
  const ms = Number(edge.invalidAt.seconds) * 1000 + edge.invalidAt.nanos / 1e6;
  return ms <= now.getTime();
}

/**
 * The whole knowledge graph read back as triples, and how many retired edges
 * were left out.
 *
 * This uses memory:inspect rather than a traversal because of edge DIRECTION. A
 * traversal row does not project an edge's endpoints, so orientation is only
 * known when a step pins a direction, and which way a fact points is the
 * model's phrasing choice: an outgoing walk from any one node silently loses
 * half the graph. Inspect returns every edge with its source and target.
 *
 * Retired edges are filtered here because inspect enumerates what is STORED,
 * superseded assertions included, where a query answers what is TRUE. Left in,
 * a correction would leave the prompt holding both "lives in Osaka" and "lives
 * in Tokyo" as current. The retired edge is still stored and readable, which is
 * what makes it history rather than a deletion; the count lets the caller say
 * so.
 */
export async function recallFacts(
  client: Client,
  agentId: string,
  o: CallOpts & { now?: Date } = {},
): Promise<{ facts: string[]; retired: number }> {
  const now = o.now ?? new Date();
  const labels = new Map<string, string>();
  const edges: GraphEdge[] = [];
  let nodeTok = "";
  let edgeTok = "";
  for (let page = 0; page < MAX_RECALL_PAGES; page++) {
    const resp = await client.memory.inspectMemory(
      {
        agentInstanceId: agentId,
        graph: {
          nodeLimit: RECALL_PAGE_LIMIT,
          edgeLimit: RECALL_PAGE_LIMIT,
          nodePageToken: nodeTok,
          edgePageToken: edgeTok,
        },
      },
      { timeoutMs: CALL_TIMEOUT_MS, signal: o.signal },
    );
    for (const n of resp.graph?.nodes ?? []) labels.set(n.nodeId, n.label);
    edges.push(...(resp.graph?.edges ?? []));
    // The two listings exhaust independently, so keep going while EITHER has
    // more. An empty token means that listing is done, not merely this page.
    nodeTok = resp.nextNodeToken;
    edgeTok = resp.nextEdgeToken;
    if (!nodeTok && !edgeTok) break;
  }

  // A node id with no label means the node listing was cut short while its
  // edges came back; the raw id is more useful to the model than dropping the
  // fact.
  const label = (nid: string): string => (labels.get(nid) ?? "").trim() || nid;

  const facts: string[] = [];
  const seen = new Set<string>();
  let retired = 0;
  for (const e of edges) {
    if (isRetired(e, now)) {
      retired++;
      continue;
    }
    const line = tripleText(label(e.sourceNodeId), e.relationshipType, label(e.targetNodeId));
    if (!seen.has(line)) {
      facts.push(line);
      seen.add(line);
    }
  }
  return { facts, retired };
}

export async function recall(client: Client, agentId: string, query: string, o: CallOpts = {}): Promise<Recall> {
  const snippets = await recallSemantic(client, agentId, query, o);
  const { facts, retired } = await recallFacts(client, agentId, o);
  return { facts, snippets, retired };
}

// ---- vocabulary ----

/**
 * The vocabulary formation will classify this workspace's memory against, for
 * the startup banner.
 *
 * READ ONLY, and not because it was simpler. Declaring a vocabulary is
 * management-class, so an operator does it out of band and the agent lives with
 * what resolves. Even the read needs agent.vocabulary:read, which the member
 * default bundle does not carry, so a refusal is reported as the ordinary
 * outcome it is. A chatbot that refused to start over it would be making the
 * wrong thing essential.
 */
export async function vocabularySummary(client: Client, agentId: string): Promise<string> {
  let resp;
  try {
    resp = await client.memory.getMemoryVocabulary({ scopeId: agentId }, { timeoutMs: CALL_TIMEOUT_MS });
  } catch (e) {
    if (code(e) === Code.PermissionDenied) {
      return (
        "not readable with this credential (it lacks the agent.vocabulary:read permission); " +
        "formation still classifies against whatever is declared"
      );
    }
    return `could not be read (${describe(e)}); formation still classifies against whatever is declared`;
  }
  // RESOLVED, not the declaration: this scope's own declaration if it has one
  // and the enterprise default otherwise, which is what a formation classifies
  // against.
  const classes = resp.resolved?.entityClasses.length ?? 0;
  const relations = resp.resolved?.relationTypes.length ?? 0;
  if (!classes && !relations) {
    return (
      "none declared, so entities are extracted untyped " +
      `(declare one with: jnh vocabulary declare --scope ${agentId} --from-file vocabulary.yaml)`
    );
  }
  return `${classes} entity class(es), ${relations} relation type(s) in effect`;
}

// ---- writes ----

/**
 * Hand the recent turns to memory:form.
 *
 * This is the whole write path of the default arm: no extraction, no node ids,
 * no direction convention, no normalizer, because the platform runs extract,
 * recall, reconcile and commit behind this one call.
 *
 * observedAt is left unset, meaning "when this formation is received", which
 * for a live conversation is when it happened. It exists for backfills, where
 * valid time is not ingest time.
 */
export function form(
  client: Client,
  agentId: string,
  turns: ConversationTurn[],
  key: string,
  o: CallOpts = {},
): Promise<FormMemoryResponse> {
  return client.memory.formMemory(
    { scopeId: agentId, turns, formationKey: key },
    { timeoutMs: FORM_TIMEOUT_MS, signal: o.signal },
  );
}

export function commit(client: Client, req: CommitMemoryRequest, o: CallOpts = {}): Promise<CommitMemoryResponse> {
  return client.memory.commitMemory(req, { timeoutMs: CALL_TIMEOUT_MS, signal: o.signal });
}
