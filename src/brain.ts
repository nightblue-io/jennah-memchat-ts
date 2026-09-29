// The pluggable chat model: Claude through @anthropic-ai/sdk, or Gemini through
// @google/genai.
//
// A brain owns the session-local conversation history, so the current chat
// stays coherent, and turns a freshly built system prompt plus the user's
// message into a reply. Long-term memory is Jennah's; nothing Jennah-facing
// depends on which brain answered.
//
// The facts a brain returns belong to the --authored arm alone. By default what
// is worth remembering is the platform's decision, so the model is offered no
// tool and the tool-call branches below never fire. That is the shape of the
// change formation makes to a client: not a different call, one fewer job.

import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenAI, Type, type Content, type GenerateContentConfig, type Part, type Tool } from "@google/genai";

import { factFromArgs, TOOL_DESC, TOOL_NAME, TOOL_PROPERTIES, TOOL_REQUIRED, type Fact } from "./authored.js";

export const ANTHROPIC_MODEL = "claude-sonnet-5-5";
// The same id works on AI Studio and on Vertex AI.
export const GEMINI_MODEL = "gemini-3.8-flash";
const MAX_TOKENS = 2048;

export interface Answer {
  reply: string;
  facts: Fact[];
}

export interface Brain {
  readonly label: string;
  chat(system: string, userMsg: string): Promise<Answer>;
}

type Env = Readonly<Record<string, string | undefined>>;

/**
 * Whether Gemini goes through Vertex AI rather than an AI Studio key: when asked
 * explicitly, or when a GCP project is set and no Studio key is.
 */
export function useVertex(env: Env): boolean {
  if (["1", "true"].includes((env.GOOGLE_GENAI_USE_VERTEXAI ?? "").toLowerCase())) return true;
  return !env.GEMINI_API_KEY && !env.GOOGLE_API_KEY && Boolean(env.GOOGLE_CLOUD_PROJECT);
}

/**
 * Resolve --provider. "auto" prefers Anthropic when an Anthropic key is
 * present, else Gemini when a Studio key or Vertex configuration is, so someone
 * with one key set needs no flag.
 */
export function selectProvider(provider: string, anthropicKey: string, env: Env): "anthropic" | "gemini" {
  const p = provider.toLowerCase();
  if (p === "auto") {
    if (anthropicKey) return "anthropic";
    if (env.GEMINI_API_KEY || env.GOOGLE_API_KEY || useVertex(env)) return "gemini";
    throw new Error(
      "no chat credentials found: set GEMINI_API_KEY or the Vertex AI env (Gemini), " +
        "or pass --anthropic-api-key / set ANTHROPIC_API_KEY (Anthropic), or pass --provider",
    );
  }
  if (p === "anthropic" || p === "claude") return "anthropic";
  if (p === "gemini") return "gemini";
  throw new Error(`unknown --provider '${provider}' (want auto|gemini|anthropic)`);
}

export function newBrain(provider: string, anthropicKey: string, offerTool: boolean, env: Env = process.env): Brain {
  if (selectProvider(provider, anthropicKey, env) === "anthropic") return new AnthropicBrain(anthropicKey, offerTool);
  return new GeminiBrain(offerTool, env);
}

export class AnthropicBrain implements Brain {
  readonly label = `anthropic/${ANTHROPIC_MODEL}`;
  private readonly client: Anthropic;
  private readonly history: Anthropic.MessageParam[] = [];
  private readonly tools: Anthropic.Tool[] = [];

  constructor(apiKey: string, offerTool: boolean, client?: Anthropic) {
    // An empty key leaves the SDK's own ANTHROPIC_API_KEY lookup in charge.
    this.client = client ?? new Anthropic(apiKey ? { apiKey } : {});
    if (offerTool) {
      this.tools.push({
        name: TOOL_NAME,
        description: TOOL_DESC,
        input_schema: {
          type: "object",
          properties: Object.fromEntries(
            Object.entries(TOOL_PROPERTIES).map(([k, d]) => [k, { type: "string", description: d }]),
          ),
          required: TOOL_REQUIRED,
        },
      });
    }
  }

  async chat(system: string, userMsg: string): Promise<Answer> {
    this.history.push({ role: "user", content: userMsg });
    const facts: Fact[] = [];
    const reply: string[] = [];
    for (;;) {
      const resp = await this.client.messages.create({
        model: ANTHROPIC_MODEL,
        max_tokens: MAX_TOKENS,
        system,
        messages: this.history,
        thinking: { type: "adaptive" },
        ...(this.tools.length ? { tools: this.tools } : {}),
      });
      // The whole content goes back into history, thinking blocks included,
      // because a tool-use turn must be continued with them intact.
      this.history.push({ role: "assistant", content: resp.content as Anthropic.ContentBlockParam[] });
      const results: Anthropic.ToolResultBlockParam[] = [];
      for (const blk of resp.content) {
        if (blk.type === "text") {
          reply.push(blk.text);
        } else if (blk.type === "tool_use") {
          const f = factFromArgs(blk.input as Record<string, unknown>);
          if (f) facts.push(f);
          results.push({ type: "tool_result", tool_use_id: blk.id, content: "noted" });
        }
      }
      if (resp.stop_reason !== "tool_use") break;
      this.history.push({ role: "user", content: results });
    }
    return { reply: reply.join("").trim(), facts };
  }
}

/** The one method GeminiBrain needs, so a test can stand in for the SDK. */
export interface GeminiModels {
  generateContent: GoogleGenAI["models"]["generateContent"];
}

export class GeminiBrain implements Brain {
  readonly label: string;
  private readonly models: GeminiModels;
  private readonly history: Content[] = [];
  private readonly tools: Tool[] | undefined;

  constructor(offerTool: boolean, env: Env, models?: GeminiModels) {
    let backend = "ai-studio";
    if (useVertex(env)) {
      // Vertex uses Application Default Credentials, not an API key
      // (gcloud auth application-default login).
      const project = env.GOOGLE_CLOUD_PROJECT ?? "";
      if (!project) {
        throw new Error(
          "Vertex AI selected but GOOGLE_CLOUD_PROJECT is not set (and run: gcloud auth application-default login)",
        );
      }
      // Gemini 3.8 is served on the global endpoint.
      const location = env.GOOGLE_CLOUD_LOCATION || env.GOOGLE_CLOUD_REGION || "global";
      backend = `vertex:${project}/${location}`;
      this.models = models ?? new GoogleGenAI({ vertexai: true, project, location }).models;
    } else {
      this.models = models ?? new GoogleGenAI({ apiKey: env.GEMINI_API_KEY || env.GOOGLE_API_KEY }).models;
    }
    if (offerTool) {
      this.tools = [
        {
          functionDeclarations: [
            {
              name: TOOL_NAME,
              description: TOOL_DESC,
              parameters: {
                type: Type.OBJECT,
                properties: Object.fromEntries(
                  Object.entries(TOOL_PROPERTIES).map(([k, d]) => [k, { type: Type.STRING, description: d }]),
                ),
                required: TOOL_REQUIRED,
              },
            },
          ],
        },
      ];
    }
    this.label = `gemini/${GEMINI_MODEL} (${backend})`;
  }

  async chat(system: string, userMsg: string): Promise<Answer> {
    // The system prompt embeds freshly recalled memory, so it is set per
    // request. Automatic function calling is off because the loop below
    // handles remember_fact itself.
    const config: GenerateContentConfig = {
      systemInstruction: system,
      maxOutputTokens: MAX_TOKENS,
      automaticFunctionCalling: { disable: true },
      ...(this.tools ? { tools: this.tools } : {}),
    };
    this.history.push({ role: "user", parts: [{ text: userMsg }] });
    const facts: Fact[] = [];
    const reply: string[] = [];
    for (;;) {
      const resp = await this.models.generateContent({ model: GEMINI_MODEL, contents: this.history, config });
      const content = resp.candidates?.[0]?.content;
      if (!content) break;
      // Kept whole, thought signatures included, because Gemini checks them
      // when a function-calling turn is continued.
      this.history.push(content);
      const results: Part[] = [];
      for (const part of content.parts ?? []) {
        if (part.functionCall) {
          const f = factFromArgs(part.functionCall.args);
          if (f) facts.push(f);
          results.push({ functionResponse: { name: part.functionCall.name, response: { result: "noted" } } });
        } else if (part.text && !part.thought) {
          reply.push(part.text);
        }
      }
      if (!results.length) break;
      // Feed the tool results back so the model can produce its final reply.
      this.history.push({ role: "user", parts: results });
    }
    return { reply: reply.join("").trim(), facts };
  }
}
