// An in-process Jennah for tests: the agent and memory services over real gRPC.
//
// Tests reach it through the SDK's own `new Client({ endpoint, insecure: true,
// apiKey })`, so every call goes through the same transport, interceptors and
// deadlines a live run does. The fake records each request with the
// grpc-timeout header that arrived with it, so a deadline test asserts what went
// over the wire rather than what the code meant to send. Each method's answer
// can be set, delayed, held open, or turned into a failure.

import { create, type DescMessage, type MessageInitShape, type MessageShape } from "@bufbuild/protobuf";
import { Code, ConnectError, type ConnectRouter, type HandlerContext } from "@connectrpc/connect";
import { connectNodeAdapter } from "@connectrpc/connect-node";
import * as http2 from "node:http2";
import type { AddressInfo } from "node:net";

import {
  AgentInstanceSchema,
  AgentService,
  CreateAgentResponseSchema,
  GetAgentResponseSchema,
} from "jennah-sdk-ts/gen/jennah/agent/v1/agent_pb";
import {
  CommitMemoryResponseSchema,
  FormMemoryResponseSchema,
  GetMemoryVocabularyResponseSchema,
  InspectMemoryResponseSchema,
  MemoryService,
  QueryMemoryResponseSchema,
  type CommitMemoryResponse,
  type FormMemoryRequest,
  type FormMemoryResponse,
  type GetMemoryVocabularyResponse,
  type InspectMemoryResponse,
  type QueryMemoryResponse,
} from "jennah-sdk-ts/gen/jennah/agent/v1/memory_pb";

export const API_KEY = "jennah_sk_test_fake";

export interface Call {
  method: string;
  request: unknown;
  /** The grpc-timeout header in milliseconds, or undefined when none was sent. */
  timeoutMs: number | undefined;
  started: number;
  ended?: number;
}

/** A grpc-timeout header value ("300000m", "5S", ...) in milliseconds. */
export function parseGrpcTimeout(v: string | null): number | undefined {
  if (!v) return undefined;
  const m = /^(\d+)([HMSmun])$/.exec(v);
  if (!m) return undefined;
  const n = Number(m[1]);
  const unit: Record<string, number> = { H: 3_600_000, M: 60_000, S: 1000, m: 1, u: 1e-3, n: 1e-6 };
  return n * unit[m[2]!]!;
}

export const msg = <D extends DescMessage>(schema: D, init?: MessageInitShape<D>): MessageShape<D> =>
  create(schema, init);

export class FakeJennah {
  endpoint = "";
  calls: Call[] = [];
  agents = new Set<string>();
  query: QueryMemoryResponse = create(QueryMemoryResponseSchema);
  /** One response per InspectMemory call, in order; the last one repeats. */
  inspectPages: InspectMemoryResponse[] = [create(InspectMemoryResponseSchema)];
  vocabulary: GetMemoryVocabularyResponse | Code = create(GetMemoryVocabularyResponseSchema);
  commit: CommitMemoryResponse = create(CommitMemoryResponseSchema);
  /**
   * form(request) -> response. Replay by formation key is the fake's job, the
   * way it is the platform's: a repeated key returns the first receipt.
   */
  form: (req: FormMemoryRequest) => FormMemoryResponse = () => create(FormMemoryResponseSchema);
  formed = new Map<string, FormMemoryResponse>();
  extractions = 0;
  delay: Record<string, number> = {};
  fail: Record<string, Code> = {};

  of(method: string): Call[] {
    return this.calls.filter((c) => c.method === method);
  }

  private async enter(method: string, request: unknown, ctx: HandlerContext): Promise<Call> {
    const call: Call = {
      method,
      request,
      timeoutMs: parseGrpcTimeout(ctx.requestHeader.get("grpc-timeout")),
      started: performance.now(),
    };
    this.calls.push(call);
    if (ctx.requestHeader.get("authorization") !== `Bearer ${API_KEY}`) {
      throw new ConnectError("the credential is invalid", Code.Unauthenticated);
    }
    const d = this.delay[method];
    if (d) await new Promise((r) => setTimeout(r, d));
    const f = this.fail[method];
    if (f !== undefined) throw new ConnectError(`${method} failed in the fake`, f);
    return call;
  }

  private leave<T>(call: Call, resp: T): T {
    call.ended = performance.now();
    return resp;
  }

  routes = (router: ConnectRouter): void => {
    router.service(AgentService, {
      getAgent: async (req, ctx) => {
        const call = await this.enter("GetAgent", req, ctx);
        if (!this.agents.has(req.agentInstanceId)) throw new ConnectError("agent not found", Code.NotFound);
        return this.leave(
          call,
          create(GetAgentResponseSchema, {
            agent: create(AgentInstanceSchema, { agentInstanceId: req.agentInstanceId }),
          }),
        );
      },
      createAgent: async (req, ctx) => {
        const call = await this.enter("CreateAgent", req, ctx);
        this.agents.add(req.agentInstanceId);
        return this.leave(
          call,
          create(CreateAgentResponseSchema, {
            agent: create(AgentInstanceSchema, { agentInstanceId: req.agentInstanceId }),
          }),
        );
      },
    });
    router.service(MemoryService, {
      queryMemory: async (req, ctx) => this.leave(await this.enter("QueryMemory", req, ctx), this.query),
      inspectMemory: async (req, ctx) => {
        const call = await this.enter("InspectMemory", req, ctx);
        const n = this.of("InspectMemory").length - 1;
        return this.leave(call, this.inspectPages[Math.min(n, this.inspectPages.length - 1)]!);
      },
      getMemoryVocabulary: async (req, ctx) => {
        const call = await this.enter("GetMemoryVocabulary", req, ctx);
        if (typeof this.vocabulary === "number") {
          throw new ConnectError("vocabulary read refused in the fake", this.vocabulary);
        }
        return this.leave(call, this.vocabulary);
      },
      commitMemory: async (req, ctx) => this.leave(await this.enter("CommitMemory", req, ctx), this.commit),
      formMemory: async (req, ctx) => {
        const call = await this.enter("FormMemory", req, ctx);
        const prior = req.formationKey ? this.formed.get(req.formationKey) : undefined;
        if (prior) return this.leave(call, prior);
        const resp = this.form(req);
        this.extractions++;
        if (req.formationKey) this.formed.set(req.formationKey, resp);
        return this.leave(call, resp);
      },
    });
  };
}

export interface Served {
  fake: FakeJennah;
  close(): Promise<void>;
}

/** Serve a fresh fake over cleartext HTTP/2 on a free loopback port. */
export async function serve(fake = new FakeJennah()): Promise<Served> {
  const server = http2.createServer(connectNodeAdapter({ routes: fake.routes }));
  const sessions = new Set<http2.ServerHttp2Session>();
  server.on("session", (s) => {
    sessions.add(s);
    s.on("close", () => sessions.delete(s));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  fake.endpoint = `127.0.0.1:${(server.address() as AddressInfo).port}`;
  return {
    fake,
    close: () =>
      new Promise<void>((resolve) => {
        // A client's pooled connection would otherwise keep close() waiting.
        for (const s of sessions) s.destroy();
        server.close(() => resolve());
      }),
  };
}
