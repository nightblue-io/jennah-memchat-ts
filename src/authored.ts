// The --authored arm: memory extracted in this client and written with memory:commit.
//
// Everything here is what memory:form replaces. It is kept, behind --authored,
// because the contrast is the most useful thing this demo can show an
// integrator: this is the apparatus you own the moment you decide to extract
// memory in your own client (node ids, a direction convention, a relationship
// normalizer), and the default arm owns none of it.

import { createHash, randomBytes } from "node:crypto";

import { create } from "jennah-sdk-ts";

import {
  CommitMemoryRequestSchema,
  ExecutionLogStepSchema,
  GraphEdgeSchema,
  GraphNodeSchema,
  GraphWriteSchema,
  VectorChunkSchema,
  type CommitMemoryRequest,
  type GraphEdge,
  type GraphNode,
} from "jennah-sdk-ts/gen/jennah/agent/v1/memory_pb";

// The authored arm's anchor: the stable node its graph is reachable from. It is
// the one node with a fixed id rather than a content hash.
//
// It is a CLIENT CONVENTION, not a platform concept, which is why the default
// arm neither seeds nor uses it. Formation extracts relationships between
// entities the conversation NAMES, so "my name is Chew" becomes facts about an
// entity called Chew and nothing links them to the person typing. The authored
// arm gets away with anchoring only because it owns the tool schema and can rule
// that an omitted subject means the user.
export const USER_NODE = "user";

// The remember_fact tool, described once and mapped into each chat SDK's tool
// type by brain.ts so the backends stay in lockstep. It stores ONE
// (subject)-[relationship]->(object) triple per call.
//
// The subject is a field, not an assumption. A tool that hardwired the user as
// the subject made every fact a spoke off one hub, and the model packed anything
// else into the two strings it had ("Gucci, Haruka, Suna-kun" as one object).
// Hence the emphasis on exactly one entity per field.
export const TOOL_NAME = "remember_fact";
export const TOOL_DESC =
  "Store ONE durable fact in long-term memory as a (subject)-[relationship]->(object) triple. " +
  "Call once per fact, and call as many times as a message needs: a fact mentioning several " +
  "entities is several calls, never one call with a list crammed into a field. Use for stable " +
  "facts worth recalling in future sessions (the user's name, preferences, job, location and " +
  "goals, and the people, organizations, teams and things they tell you about, including how " +
  "those relate to each other); do NOT store transient chit-chat or questions.";
export const TOOL_PROPERTIES: Record<string, string> = {
  subject:
    "the single entity the fact is about, e.g. 'Alice', 'NightBlue', 'FinOps Consulting'. " +
    "OMIT it whenever the fact is the user talking about themselves ('my name is X', 'I live " +
    "in Y', 'I work at Z'), and keep omitting it once you know their name: the user already " +
    "has a dedicated node, so naming them here creates a duplicate of them. Name a subject " +
    "only for facts about someone or something else.",
  relationship:
    "short verb phrase linking subject to object, e.g. 'is named', 'likes', 'lives in', " +
    "'works at', 'has cto', 'owns', 'reports to', 'has member'",
  object:
    "the single entity or value the relationship points at, e.g. 'Alice', 'hiking', 'Tokyo', " +
    "'NightBlue'. Exactly one, never a list: three members of a team is three calls, and two " +
    "roles held by one person is two calls.",
};
// subject is optional: omitted means the user, which keeps the common case ("my
// name is Sabrina") a two-field call.
export const TOOL_REQUIRED = ["relationship", "object"];

// The instruction the authored arm adds to the system prompt. The default arm
// has no equivalent, see buildSystemPrompt in main.ts.
export const STORE_INSTRUCTION =
  "Whenever the user shares a durable fact, call remember_fact to store it as one " +
  "(subject, relationship, object) triple: their name, preferences, job, location and goals, " +
  "and also the people, organizations and teams they mention and how those relate to one " +
  "another. One call per fact, one entity per field: a team with three members is three " +
  "calls, not one call listing three names. Do not store transient chit-chat.";

/**
 * One (subject)-[relationship]->(object) triple the model chose to store. An
 * empty subject means the user.
 */
export interface Fact {
  subj: string;
  rel: string;
  obj: string;
}

/** A tool call's arguments as a Fact, or undefined when a required field is blank. */
export function factFromArgs(args: Record<string, unknown> | null | undefined): Fact | undefined {
  const str = (v: unknown): string => (v === undefined || v === null ? "" : String(v));
  const rel = str(args?.relationship);
  const obj = str(args?.object);
  if (!rel.trim() || !obj.trim()) return undefined;
  return { subj: str(args?.subject), rel, obj };
}

// INVERSE_REL canonicalizes edge DIRECTION. A key is a relationship the model
// emits pointing the "wrong" way; its value is the canonical relationship to
// store once source and target are swapped. So "Chew is CTO of NightBlue" and
// "NightBlue has CTO Chew" both land as NightBlue -[HAS_CTO]-> Chew, one edge
// with one id.
//
// The convention is container first: the organization or owner is the source,
// and the person or part it contains is the target.
//
// This is NOT an ontology. A predicate absent from this table is stored exactly
// as the model phrased it. The table lists only pairs observed being emitted
// BOTH ways across repeated runs of the same conversation.
export const INVERSE_REL: Readonly<Record<string, string>> = {
  IS_CEO_OF: "HAS_CEO",
  IS_CTO_OF: "HAS_CTO",
  IS_COO_OF: "HAS_COO",
  IS_CFO_OF: "HAS_CFO",
  IS_MEMBER_OF: "HAS_MEMBER",
  IS_PART_OF: "HAS_PART",
  IS_A_DEPARTMENT_OF: "HAS_DEPARTMENT",
  IS_OWNED_BY: "OWNS",
  BELONGS_TO: "HAS_MEMBER",
};

export function shortHash(s: string): string {
  return createHash("sha1").update(s, "utf8").digest("hex").slice(0, 12);
}

export function randId(prefix: string): string {
  return `${prefix}_${randomBytes(16).toString("hex")}`;
}

export function isUserRef(label: string): boolean {
  return ["", "user", "the user", "me", "i", "myself"].includes(label.trim().toLowerCase());
}

/**
 * The stable node id for an entity label, content-hashed so the same entity
 * named twice converges on one node. Anything that names the user folds onto
 * the fixed anchor, so a phrasing choice cannot strand facts on a rival node.
 */
export function nodeId(label: string): string {
  if (isUserRef(label)) return USER_NODE;
  return "n_" + shortHash(label.trim().toLowerCase());
}

function subjectLabel(subj: string): string {
  return isUserRef(subj) ? "user" : subj.trim();
}

/** A verb phrase as an edge RelationshipType, e.g. "is named" -> "IS_NAMED". */
export function normRel(s: string): string {
  const out = Array.from(s.trim().toUpperCase(), (c) => (/^[A-Z0-9]$/.test(c) ? c : "_"))
    .join("")
    .replace(/^_+|_+$/g, "");
  return out || "RELATED_TO";
}

/** A stored RelationshipType back as a readable phrase. */
export function prettyRel(s: string): string {
  return s ? s.replaceAll("_", " ").toLowerCase() : "->";
}

export function tripleText(subj: string, rel: string, obj: string): string {
  return `${subj} ${prettyRel(rel)} ${obj}`;
}

/**
 * The user anchor, written on every authored start.
 *
 * Every start rather than once at bootstrap, because the arm can change between
 * runs: a workspace created by the default arm has no anchor, and an authored
 * commit naming an absent node is rejected. The write is an idempotent upsert
 * of a fixed label, so re-sending it cannot drift.
 */
export function seedRequest(agentId: string): CommitMemoryRequest {
  return create(CommitMemoryRequestSchema, {
    agentInstanceId: agentId,
    graph: { nodes: [{ nodeId: USER_NODE, label: "User" }] },
  });
}

/**
 * One turn as a single CommitMemory: the exchange as a vector chunk, a log step,
 * and the facts as graph nodes and edges, written atomically.
 *
 * Returns the request and the facts as readable triples, for --verbose.
 */
export function commitRequest(
  agentId: string,
  userMsg: string,
  reply: string,
  facts: Fact[],
): { req: CommitMemoryRequest; stored: string[] } {
  const nodes: GraphNode[] = [];
  const edges: GraphEdge[] = [];
  const seenNodes = new Set<string>();
  const seenEdges = new Set<string>();

  const addNode = (label: string): string => {
    const nid = nodeId(label);
    // The anchor carries its own label from the seed; re-writing it here would
    // overwrite "User" with whatever the model typed.
    if (nid !== USER_NODE && !seenNodes.has(nid)) {
      nodes.push(create(GraphNodeSchema, { nodeId: nid, label: label.trim() }));
      seenNodes.add(nid);
    }
    return nid;
  };

  const stored: string[] = [];
  for (const f of facts) {
    // Both endpoints get a node whichever way the edge ends up pointing, so the
    // flip below only reorients the edge.
    let src = addNode(f.subj);
    let dst = addNode(f.obj);
    let srcLabel = subjectLabel(f.subj);
    let dstLabel = f.obj.trim();
    let rel = normRel(f.rel);
    const inverse = INVERSE_REL[rel];
    if (inverse) {
      [src, dst] = [dst, src];
      [srcLabel, dstLabel] = [dstLabel, srcLabel];
      rel = inverse;
    }
    // Keyed on the canonical ids and the normalized relationship, so the same
    // fact phrased differently converges on one edge.
    const eid = "e_" + shortHash(`${src}|${rel}|${dst}`);
    // A mutation set cannot carry two writes for the same key, so dedup within
    // this commit. Idempotency across commits is the server's job.
    if (!seenEdges.has(eid)) {
      edges.push(create(GraphEdgeSchema, { edgeId: eid, sourceNodeId: src, targetNodeId: dst, relationshipType: rel }));
      seenEdges.add(eid);
    }
    stored.push(tripleText(srcLabel, rel, dstLabel));
  }

  const req = create(CommitMemoryRequestSchema, {
    agentInstanceId: agentId,
    log: create(ExecutionLogStepSchema, {
      stepId: randId("step"),
      thoughtProcess: "conversation turn",
      toolUsed: "memchat",
      toolInput: userMsg.slice(0, 500),
      toolOutput: reply.slice(0, 1000),
    }),
    vectors: [create(VectorChunkSchema, { chunkId: randId("chunk"), rawContent: `User: ${userMsg}\nAssistant: ${reply}` })],
  });
  if (nodes.length || edges.length) req.graph = create(GraphWriteSchema, { nodes, edges });
  return { req, stored };
}
