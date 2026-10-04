// The pluggable chat model: Claude through @anthropic-ai/sdk (directly or on
// Amazon Bedrock), or Gemini through @google/genai.
//
// A brain owns the session-local conversation history, so the current chat
// stays coherent, and turns the persona, freshly recalled memory and the user's
// message into a reply. Long-term memory is Jennah's; nothing Jennah-facing
// depends on which brain answered.
//
// The facts a brain returns belong to the --authored arm alone. By default what
// is worth remembering is the platform's decision, so the model is offered no
// tool and the tool-call branches below never fire. That is the shape of the
// change formation makes to a client: not a different call, one fewer job.

import Anthropic from "@anthropic-ai/sdk";
import { AnthropicBedrock } from "@anthropic-ai/bedrock-sdk";
import { fromIni } from "@aws-sdk/credential-providers";
import { GoogleGenAI, Type, type Content, type GenerateContentConfig, type Part, type Tool } from "@google/genai";

import { factFromArgs, TOOL_DESC, TOOL_NAME, TOOL_PROPERTIES, TOOL_REQUIRED, type Fact } from "./authored.js";

export const ANTHROPIC_MODEL = "claude-sonnet-5-5";
// The same model on Amazon Bedrock. It is a cross-region inference profile id,
// not a bare model id: Bedrock serves current Claude models only through a
// profile, and "global." routes to whichever region has capacity at no premium
// over a single-region profile.
export const BEDROCK_MODEL = "global.anthropic.claude-sonnet-5-5";
export const DEFAULT_AWS_REGION = "ap-northeast-1";
// The same id works on AI Studio and on Vertex AI.
export const GEMINI_MODEL = "gemini-3.8-flash";
const MAX_TOKENS = 2048;

export interface Answer {
  reply: string;
  facts: Fact[];
}

export interface Brain {
  readonly label: string;
  /**
   * persona is the fixed instruction (the same every turn of a session) and
   * recall is this turn's remembered context. They arrive separately because
   * the Claude brain must keep the first unchanged and append the second.
   */
  chat(persona: string, recall: string, userMsg: string): Promise<Answer>;
}

/** Where --provider bedrock sends requests, and the named profile that signs them. */
export interface AwsTarget {
  region: string;
  profile: string;
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
 *
 * "bedrock" (Claude on Amazon Bedrock) is never chosen by "auto": AWS
 * credentials are present in many shells for reasons that have nothing to do
 * with this demo, so having them is no sign of intent.
 */
export function selectProvider(provider: string, anthropicKey: string, env: Env): "anthropic" | "bedrock" | "gemini" {
  const p = provider.toLowerCase();
  if (p === "auto") {
    if (anthropicKey) return "anthropic";
    if (env.GEMINI_API_KEY || env.GOOGLE_API_KEY || useVertex(env)) return "gemini";
    throw new Error(
      "no chat credentials found: pass --anthropic-api-key / set ANTHROPIC_API_KEY (Anthropic), " +
        "pass --provider bedrock (Claude on Amazon Bedrock, explicit only), " +
        "or set the Vertex AI env / GEMINI_API_KEY (Gemini)",
    );
  }
  if (p === "anthropic" || p === "claude") return "anthropic";
  if (p === "bedrock") return "bedrock";
  if (p === "gemini") return "gemini";
  throw new Error(`unknown --provider '${provider}' (want auto|anthropic|bedrock|gemini)`);
}

export function newBrain(
  provider: string,
  anthropicKey: string,
  offerTool: boolean,
  env: Env = process.env,
  aws: AwsTarget = { region: DEFAULT_AWS_REGION, profile: "" },
): Brain {
  switch (selectProvider(provider, anthropicKey, env)) {
    case "anthropic":
      return new AnthropicBrain(anthropicKey, offerTool);
    case "bedrock":
      return bedrockBrain(offerTool, aws);
    default:
      return new GeminiBrain(offerTool, env);
  }
}

/**
 * Claude on Amazon Bedrock: the Anthropic brain with a Bedrock client, whose
 * requests are signed by AWS credentials instead of an Anthropic key.
 *
 * A named profile is resolved from that profile alone rather than left to
 * AWS_PROFILE, because the default chain reads AWS_ACCESS_KEY_ID first: with
 * bare keys exported, AWS_PROFILE is silently ignored and the calls run, without
 * error, in whatever account those keys belong to.
 */
export function bedrockBrain(offerTool: boolean, aws: AwsTarget): AnthropicBrain {
  if (!aws.region) throw new Error("--provider bedrock needs an AWS region: pass --aws-region");
  const client = new AnthropicBedrock({
    awsRegion: aws.region,
    ...(aws.profile ? { providerChainResolver: async () => fromIni({ profile: aws.profile }) } : {}),
  });
  return new AnthropicBrain("", offerTool, client, { model: BEDROCK_MODEL, via: "bedrock" });
}

/** The one resource AnthropicBrain needs, shared by the direct and Bedrock clients. */
export interface AnthropicMessages {
  messages: Pick<Anthropic["messages"], "create">;
}

export class AnthropicBrain implements Brain {
  readonly label: string;
  private readonly client: AnthropicMessages;
  private readonly model: string;
  private readonly history: Anthropic.MessageParam[] = [];
  private readonly tools: Anthropic.Tool[] = [];

  constructor(
    apiKey: string,
    offerTool: boolean,
    client?: AnthropicMessages,
    { model = ANTHROPIC_MODEL, via = "anthropic" }: { model?: string; via?: string } = {},
  ) {
    // An empty key leaves the SDK's own ANTHROPIC_API_KEY lookup in charge.
    this.client = client ?? new Anthropic(apiKey ? { apiKey } : {});
    this.model = model;
    this.label = `${via}/${model}`;
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

  /**
   * persona goes out as the top-level system prompt, which never changes during
   * a session, and the turn's recall as a system MESSAGE right after the user's.
   *
   * Rebuilding the top-level prompt each turn with fresh recall is the obvious
   * shape and the wrong one. A thinking block is bound to the exact conversation
   * prefix it was produced under, the system prompt included, and an
   * organization created on or after 2026-08-31 is refused (400) when an earlier
   * block is replayed under a different prefix. So turn 2 of such a session
   * failed, on Bedrock and on the direct API alike, while older organizations
   * never saw it. Appending recall instead edits nothing that came before: every
   * earlier block stays valid, and the unchanged prefix is cacheable as a bonus.
   *
   * Old recall stays in the transcript, so a long session carries every
   * snapshot (more input tokens per turn). A mid-conversation system message
   * needs a model that accepts one: Sonnet 5.5 does; Sonnet 5 answers 400
   * "role 'system' is not supported", so swapping the model to it means moving
   * recall into the user turn.
   */
  async chat(persona: string, recall: string, userMsg: string): Promise<Answer> {
    this.history.push({ role: "user", content: userMsg }, { role: "system", content: recall });
    const facts: Fact[] = [];
    const reply: string[] = [];
    for (;;) {
      const resp = await this.client.messages.create({
        model: this.model,
        max_tokens: MAX_TOKENS,
        system: persona,
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

  async chat(persona: string, recall: string, userMsg: string): Promise<Answer> {
    // The system instruction embeds freshly recalled memory, so it is set per
    // request. Gemini does not bind its history to the instruction it was
    // produced under, so unlike the Claude brain this needs no append-only
    // shape. Automatic function calling is off because the loop below handles
    // remember_fact itself.
    const config: GenerateContentConfig = {
      systemInstruction: `${persona}\n\n${recall}`,
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
