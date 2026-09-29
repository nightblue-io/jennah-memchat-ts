// memchat: a chatbot that remembers across sessions, built on Jennah's
// TypeScript SDK.
//
// Each turn does recall, answer, form:
//
// 1. recall: memory:query for semantic recall of past exchanges (the platform
//    embeds the query text), plus memory:inspect to read the knowledge graph
//    back as triples.
// 2. answer: the chat model answers the user, and that is ALL it is asked for.
//    It is given no memory tool and no instruction about what to remember.
// 3. form: memory:form extracts candidate memories from the recent turns,
//    recalls what the workspace already holds, reconciles the two, and commits
//    the result atomically. The receipt says what it decided about every
//    candidate, and what it RETIRED to make room for a correction.
//
// --authored is the other arm: the model is handed a remember_fact tool, and
// this client turns the triples it emits into graph nodes and edges itself and
// writes them with memory:commit. Both arms write into the same workspace,
// since formed and authored memory are the same rows. Run one, then the other,
// and the difference is the cognition layer.
//
// Cross-session memory is simply reusing the same workspace id, persisted to a
// small state file.

import * as readline from "node:readline";
import { parseArgs } from "node:util";

import { Client, create, DEFAULT_ENDPOINT, JennahError, NoCredentialError } from "jennah-sdk-ts";
import { ConversationTurnSchema, TurnRole, type ConversationTurn } from "jennah-sdk-ts/gen/jennah/agent/v1/memory_pb";

import { commitRequest, randId, seedRequest, STORE_INSTRUCTION, type Fact } from "./authored.js";
import { newBrain, type Brain } from "./brain.js";
import * as jennah from "./jennah.js";
import { commitLines, formationLines, type Line, type Style } from "./receipt.js";

// How many recent turns each formation submits: the current exchange plus the
// two before it.
//
// Not just the current exchange, because extraction runs BEFORE recall inside a
// formation, so nothing the workspace already holds can resolve "she", "there"
// or "the second one": only the submitted turns can. The cost of the overlap is
// that already-formed content is re-extracted, which reconciliation reports as
// KNOWN instead of writing twice. That is a token bill, not a correctness
// problem, and it is why the number is small.
export const FORM_WINDOW = 6;

export type Out = (style: Style | "plain" | "err", text: string) => void;

const COLORS: Record<Style, string> = { dim: "\x1b[2m", cyan: "\x1b[36m", yellow: "\x1b[33m" };

/**
 * An Out that prints to stdout, colored when stdout is a terminal. "err" goes
 * to stderr.
 */
export function consoleOut(stream: NodeJS.WriteStream = process.stdout): Out {
  const color = stream.isTTY && !process.env.NO_COLOR;
  return (style, text) => {
    if (style === "err") process.stderr.write(text + "\n");
    else if (style === "plain") stream.write(text + "\n");
    else if (color) stream.write(`  ${COLORS[style]}${text}\x1b[0m\n`);
    else stream.write(`  ${text}\n`);
  };
}

/** The last FORM_WINDOW turns: the slice each formation submits. */
export function window<T>(turns: T[]): T[] {
  return turns.slice(-FORM_WINDOW);
}

/**
 * The name of one formation, so that resending it is safe.
 *
 * It matters more here than an idempotency key usually does. Extraction is
 * nondeterministic, so a blind resend does not repeat the first attempt: it
 * forms a second, different set of memory. Under this key a resend replays the
 * original receipt and extracts nothing.
 *
 * A formation here is (this workspace, this session, this turn ordinal). The
 * session id keeps the first turn of every session from being the same
 * formation, and the ordinal keeps a user who says "thanks" twice from losing
 * the second one to a replay, which hashing the text would do.
 */
export function formationKey(sessionId: string, turnNo: number): string {
  return `frm_${sessionId}_${turnNo}`;
}

/**
 * The turn's system prompt, from what Jennah recalled.
 *
 * The instruction about STORING memory appears only in the authored arm, and
 * its absence by default is not a simplification: a prompt telling the model
 * what to remember while the platform independently decides the same thing
 * would be two extractors with one workspace, disagreeing at the caller's
 * expense.
 */
export function buildSystemPrompt(rec: jennah.Recall, authored: boolean): string {
  const parts = [
    "You are Memo, a warm, concise assistant with long-term memory that persists across sessions. " +
      "Personalize using the remembered context below and refer back to it naturally. ",
  ];
  if (authored) parts.push(STORE_INSTRUCTION);
  parts.push("\n\n# What you already know (knowledge graph)\n");
  if (rec.facts.length) parts.push(...rec.facts.map((f) => `- ${f}\n`));
  else parts.push("(nothing yet, this may be your first conversation)\n");
  parts.push("\n# Relevant snippets from past conversations\n");
  if (rec.snippets.length) parts.push(...rec.snippets.map((s) => `- ${s.text}\n`));
  else parts.push("(none retrieved)\n");
  return parts.join("");
}

export function retiredNote(n: number): string {
  return n ? ` (+${n} retired, not shown)` : "";
}

/**
 * One session: recall, answer, then write, per line of input.
 *
 * transcript is the formation arm's own copy of the recent turns, in the wire
 * type memory:form takes. The brain's history cannot serve: it is in whichever
 * vendor SDK's message type answered.
 */
export class Chat {
  readonly transcript: ConversationTurn[] = [];
  readonly sessionId = randId("sess");
  turnNo = 0;

  constructor(
    private readonly client: Client,
    private readonly brain: Brain,
    private readonly agentId: string,
    private readonly opts: { authored: boolean; verbose: boolean; out: Out },
  ) {}

  private emit(lines: Line[]): void {
    for (const [style, text] of lines) this.opts.out(style, text);
  }

  async turn(line: string, signal?: AbortSignal): Promise<void> {
    const { out, verbose, authored } = this.opts;
    let rec: jennah.Recall;
    try {
      rec = await jennah.recall(this.client, this.agentId, line, { signal });
    } catch (e) {
      if (!signal?.aborted) out("err", `error: recall: ${jennah.describe(e)}`);
      return;
    }
    const summary =
      `recalled ${rec.facts.length} fact(s)${retiredNote(rec.retired)}, ` + `${rec.snippets.length} past snippet(s)`;
    if (verbose) {
      out("dim", summary + ":");
      for (const f of rec.facts) out("dim", `  - ${f}`);
      for (const s of rec.snippets) out("dim", `  ~ ${s.text}${s.prov}`);
    } else {
      out("dim", `[${summary}]`);
    }

    let reply: string;
    let facts: Fact[];
    try {
      ({ reply, facts } = await this.brain.chat(buildSystemPrompt(rec, authored), line));
    } catch (e) {
      // The vendor SDKs throw their own types.
      if (!signal?.aborted) out("err", `error: chat model: ${jennah.describe(e)}`);
      return;
    }
    out("plain", `\nmemo> ${reply}`);

    // The reply is on screen BEFORE memory is written. A formation runs model
    // inference and a retrieval before it writes, so it is a seconds-class call
    // by contract, and making the user wait on it to read an answer the model
    // already produced would be self-inflicted latency.
    this.turnNo++;
    this.transcript.push(
      create(ConversationTurnSchema, { role: TurnRole.USER, content: line }),
      create(ConversationTurnSchema, { role: TurnRole.ASSISTANT, content: reply }),
    );
    if (authored) await this.commit(line, reply, facts, signal);
    else await this.form(signal);
  }

  private async form(signal?: AbortSignal): Promise<void> {
    const { out, verbose } = this.opts;
    const turns = window(this.transcript);
    const key = formationKey(this.sessionId, this.turnNo);
    if (verbose) out("dim", `forming memory from ${turns.length} turn(s), key ${key} ...`);
    else out("dim", "[forming memory ...]");
    let resp;
    try {
      resp = await jennah.form(this.client, this.agentId, turns, key, { signal });
    } catch (e) {
      out("err", `warning: could not form this turn's memory: ${jennah.describe(e)}`);
      return;
    }
    this.emit(formationLines(resp, verbose));
  }

  private async commit(line: string, reply: string, facts: Fact[], signal?: AbortSignal): Promise<void> {
    const { out, verbose } = this.opts;
    const { req, stored } = commitRequest(this.agentId, line, reply, facts);
    let resp;
    try {
      resp = await jennah.commit(this.client, req, { signal });
    } catch (e) {
      out("err", `warning: could not persist this turn's memory: ${jennah.describe(e)}`);
      return;
    }
    if (verbose) for (const s of stored) out("dim", `stored fact: ${s}`);
    this.emit(commitLines(resp, verbose));
  }
}

const OPTIONS = {
  endpoint: { type: "string" },
  insecure: { type: "boolean", default: false },
  state: { type: "string", default: "memchat-state.json" },
  agent: { type: "string", default: "" },
  provider: { type: "string", default: "auto" },
  region: { type: "string" },
  "jennah-api-key": { type: "string", default: "" },
  "anthropic-api-key": { type: "string", default: "" },
  verbose: { type: "boolean", default: false },
  authored: { type: "boolean", default: false },
  help: { type: "boolean", short: "h", default: false },
} as const;

// Written by hand, and no secret is ever a default, so --help cannot print one.
// The SDK and the vendor SDKs read their env vars themselves when a flag is not
// passed.
export const HELP = `usage: memchat-ts [flags]

A chatbot that remembers across sessions, with memory kept in Jennah.

flags:
  --endpoint HOST:PORT       Jennah gRPC endpoint (default ${DEFAULT_ENDPOINT})
  --insecure                 connect without TLS, for a local plaintext server
  --state PATH               local state file holding the workspace id
                             (default memchat-state.json)
  --agent ID                 use this EXISTING agent workspace instead of the one
                             in the state file, for a workspace provisioned out of
                             band (e.g. one with a vocabulary declared on it).
                             Never creates and never writes the state file
  --provider NAME            chat model: auto|gemini|anthropic (default auto:
                             Anthropic if its key is set, else Gemini)
  --region REGION            home region for a NEW workspace (e.g. us-central1),
                             also read from $JENNAH_REGION; empty uses the platform
                             default. List regions with 'jnh agents regions'
  --jennah-api-key KEY       Jennah API key (jennah_sk_...); falls back to
                             $JENNAH_API_KEY, then 'jnh login'
  --anthropic-api-key KEY    Anthropic API key; falls back to $ANTHROPIC_API_KEY
  --verbose                  print recalled memory and full receipts each turn
  --authored                 extract and author the memory writes in this client
                             (remember_fact + memory:commit) instead of letting
                             memory:form do it. Writes into the same workspace
  -h, --help                 show this help
`;

export interface Args {
  endpoint?: string;
  insecure: boolean;
  state: string;
  agent: string;
  provider: string;
  region: string;
  jennahApiKey: string;
  anthropicApiKey: string;
  verbose: boolean;
  authored: boolean;
  help: boolean;
}

export function parseFlags(argv: string[], env: NodeJS.ProcessEnv = process.env): Args {
  const { values: v } = parseArgs({ args: argv, options: OPTIONS, strict: true, allowPositionals: false });
  return {
    endpoint: v.endpoint,
    insecure: v.insecure,
    state: v.state,
    agent: v.agent,
    provider: v.provider,
    // The region is not a secret, so its env var may stand in for the flag.
    region: v.region ?? env.JENNAH_REGION ?? "",
    jennahApiKey: v["jennah-api-key"],
    anthropicApiKey: v["anthropic-api-key"],
    verbose: v.verbose,
    authored: v.authored,
    help: v.help,
  };
}

/**
 * The SDK client, with the credential resolved the SDK's way: the flag when
 * given, else $JENNAH_API_KEY, else the 'jnh login' session.
 */
export function connect(args: Args): Client {
  try {
    return new Client({
      apiKey: args.jennahApiKey.trim() || undefined,
      endpoint: args.endpoint,
      insecure: args.insecure,
    });
  } catch (e) {
    if (e instanceof NoCredentialError) {
      throw new jennah.StartupError(
        "no Jennah credential found: pass --jennah-api-key, set JENNAH_API_KEY, or run 'jnh login'",
      );
    }
    if (e instanceof JennahError) throw new jennah.StartupError(e.message);
    throw e;
  }
}

export interface Deps {
  out?: Out;
  input?: NodeJS.ReadableStream;
  output?: NodeJS.WritableStream;
  env?: NodeJS.ProcessEnv;
  /** Replaces the chat model, for tests. */
  brain?: (provider: string, anthropicKey: string, offerTool: boolean) => Brain;
}

export async function main(argv: string[], deps: Deps = {}): Promise<number> {
  const out = deps.out ?? consoleOut();
  const env = deps.env ?? process.env;
  let args: Args;
  try {
    args = parseFlags(argv, env);
  } catch (e) {
    out("err", `memchat-ts: ${(e as Error).message} (see --help)`);
    return 2;
  }
  if (args.help) {
    (deps.output ?? process.stdout).write(HELP);
    return 0;
  }

  let client: Client | undefined;
  let brain: Brain;
  let agentId: string;
  try {
    // The credential is checked first, so a missing one fails before anything
    // touches the network.
    client = connect(args);
    const anthropicKey = args.anthropicApiKey.trim() || env.ANTHROPIC_API_KEY || "";
    try {
      brain = (deps.brain ?? ((p, k, t) => newBrain(p, k, t, env)))(args.provider, anthropicKey, args.authored);
    } catch (e) {
      throw new jennah.StartupError((e as Error).message);
    }
    out("plain", `chat model: ${brain.label}`);
    out(
      "plain",
      args.authored
        ? "memory: authored in this client (remember_fact + memory:commit)"
        : "memory: formed by Jennah (memory:form)",
    );
    let where: string;
    try {
      ({ agentId, where } = await jennah.resolveWorkspace(client, {
        agent: args.agent,
        statePath: args.state,
        region: args.region,
      }));
    } catch (e) {
      if (e instanceof jennah.StartupError) throw e;
      throw new jennah.StartupError(`workspace: ${jennah.describe(e)}`);
    }
    out("plain", where);
    if (args.authored) {
      try {
        await jennah.commit(client, seedRequest(agentId));
      } catch (e) {
        throw new jennah.StartupError(`seed user node: ${jennah.describe(e)}`);
      }
    } else {
      out("plain", `vocabulary: ${await jennah.vocabularySummary(client, agentId)}`);
    }
  } catch (e) {
    client?.close();
    if (e instanceof jennah.StartupError) {
      out("err", `memchat-ts: ${e.message}`);
      return 1;
    }
    throw e;
  }

  out("plain", "\nmemchat: a chatbot that remembers across sessions (Ctrl-D or /exit to quit).");
  out("plain", "Tell it about yourself, quit, run it again, and it'll recall.");
  const chat = new Chat(client, brain, agentId, { authored: args.authored, verbose: args.verbose, out });

  const input = deps.input ?? process.stdin;
  const output = deps.output ?? process.stdout;
  const rl = readline.createInterface({ input, output, terminal: (input as NodeJS.ReadStream).isTTY === true });
  rl.setPrompt("you> ");
  // The interface closes when input ends (Ctrl-D, or the end of piped input)
  // while lines already read are still being answered, and prompting after
  // that throws.
  let open = true;
  rl.on("close", () => {
    open = false;
  });
  const prompt = (): void => {
    if (!open) return;
    output.write("\n");
    rl.prompt();
  };
  // Ctrl-C ends the session, and aborts whatever the current turn is waiting on.
  let inFlight: AbortController | undefined;
  rl.on("SIGINT", () => {
    inFlight?.abort();
    rl.close();
  });
  try {
    prompt();
    for await (const raw of rl) {
      const line = raw.trim();
      if (line === "/exit" || line === "/quit") break;
      if (line) {
        inFlight = new AbortController();
        await chat.turn(line, inFlight.signal);
        inFlight = undefined;
      }
      prompt();
    }
  } finally {
    rl.close();
    client.close();
  }
  out("plain", "\nbye. Your memory is saved in Jennah.");
  return 0;
}
