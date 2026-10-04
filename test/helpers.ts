// Shared by the tests: a recording Out, a scripted brain, an empty machine, and
// a fake Jennah with a client pointed at it.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { PassThrough, Readable } from "node:stream";

import { Client } from "jennah-sdk-ts";

import type { Fact } from "../src/authored.js";
import type { Answer, Brain } from "../src/brain.js";
import { Chat, type Out } from "../src/main.js";
import { API_KEY, serve, type FakeJennah } from "./fake.js";

/** An Out that records [style, text], and a timeline tests can append to. */
export class Events {
  lines: [string, string][] = [];
  timeline: [string, string][] = [];
  out: Out = (style, text) => {
    this.lines.push([style, text]);
    this.timeline.push(["out", text]);
  };
  text(): string {
    return this.lines.map(([, t]) => t).join("\n");
  }
}

export class FakeBrain implements Brain {
  readonly label = "fake/brain";
  systems: string[] = [];
  personas: string[] = [];
  constructor(
    public reply = "ok",
    public facts: Fact[] = [],
  ) {}
  async chat(persona: string, recall: string): Promise<Answer> {
    this.personas.push(persona);
    this.systems.push(`${persona}\n\n${recall}`);
    return { reply: this.reply, facts: [...this.facts] };
  }
}

export interface Machine {
  dir: string;
  sessionPath: string;
  restore(): void;
}

/**
 * An empty machine: a fresh per-user config directory and no API key in the
 * environment, so neither the developer's real session nor their key leaks in.
 */
export function emptyMachine(): Machine {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "memchat-ts-"));
  const saved = new Map<string, string | undefined>();
  for (const k of ["XDG_CONFIG_HOME", "HOME", "APPDATA", "JENNAH_API_KEY"]) saved.set(k, process.env[k]);
  process.env.XDG_CONFIG_HOME = dir;
  process.env.HOME = dir;
  process.env.APPDATA = dir;
  delete process.env.JENNAH_API_KEY;
  return {
    dir,
    sessionPath: path.join(dir, "jennah", "credentials"),
    restore() {
      for (const [k, v] of saved) {
        if (v === undefined) delete process.env[k];
        else process.env[k] = v;
      }
      fs.rmSync(dir, { recursive: true, force: true });
    },
  };
}

export function tmpDir(): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), "memchat-ts-"));
}

export interface World {
  fake: FakeJennah;
  client: Client;
  close(): Promise<void>;
}

/** A fake Jennah and a client that reaches it the way a live run would. */
export async function world(): Promise<World> {
  const served = await serve();
  const client = new Client({ endpoint: served.fake.endpoint, insecure: true, apiKey: API_KEY });
  return {
    fake: served.fake,
    client,
    close: async () => {
      client.close();
      await served.close();
    },
  };
}

export function chat(
  w: World,
  o: { authored?: boolean; verbose?: boolean; brain?: FakeBrain } = {},
): { c: Chat; ev: Events; brain: FakeBrain } {
  w.fake.agents.add("demo.a");
  const ev = new Events();
  const brain = o.brain ?? new FakeBrain();
  const c = new Chat(w.client, brain, "demo.a", { authored: !!o.authored, verbose: !!o.verbose, out: ev.out });
  return { c, ev, brain };
}

/** Standard input that ends at once, as Ctrl-D would, and a sink for prompts. */
export function noInput(): { input: Readable; output: PassThrough } {
  const output = new PassThrough();
  output.resume();
  return { input: Readable.from([]), output };
}
