// Rendering what a formation or a commit reported.
//
// Each function returns lines as [style, text] pairs and prints nothing, so the
// wording is testable on its own and the caller decides about color. style is
// one of "dim", "cyan" or "yellow".

import {
  CandidateKind,
  MemoryDecision,
  type CommitMemoryResponse,
  type FormedCandidate,
  type FormMemoryResponse,
} from "jennah-sdk-ts/gen/jennah/agent/v1/memory_pb";

import { tripleText } from "./authored.js";

export type Style = "dim" | "cyan" | "yellow";
export type Line = [Style, string];

const DECISIONS = [MemoryDecision.NEW, MemoryDecision.REVISED, MemoryDecision.KNOWN, MemoryDecision.REJECTED];

export function singleLine(s: string): string {
  return s.split(/\s+/).filter(Boolean).join(" ");
}

/** The short label for a decision, e.g. MemoryDecision.NEW -> "new". */
export function decisionWord(d: MemoryDecision): string {
  if (d === MemoryDecision.UNSPECIFIED) return "?";
  return (MemoryDecision[d] ?? String(d)).toLowerCase();
}

/**
 * A candidate the way it will be remembered: a relationship as its triple,
 * anything else as the text that gets stored.
 */
export function candidateText(c: FormedCandidate): string {
  if (c.kind === CandidateKind.RELATIONSHIP && c.sourceEntity) {
    return tripleText(c.sourceEntity, c.relationshipType, c.targetEntity);
  }
  return singleLine(c.text);
}

/**
 * The part of a decision that only means something next to its subject: what a
 * revision retired, what a known candidate matched, and why a rejected one was
 * refused. The rejection reason is how a caller learns that a conversation
 * recounting history was not allowed to retire the current fact.
 */
export function decisionNote(c: FormedCandidate): string {
  if (c.decision === MemoryDecision.REVISED && c.matchedId) return `  (retired ${c.matchedId})`;
  if (c.decision === MemoryDecision.KNOWN && c.matchedId) return `  (matches ${c.matchedId})`;
  if (c.decision === MemoryDecision.REJECTED && c.rejectionReason.trim()) {
    return `  (${singleLine(c.rejectionReason)})`;
  }
  return "";
}

/**
 * The one-line count. Zero candidates is a legitimate outcome, not a failure: a
 * turn can hold nothing worth remembering, and saying so beats "formed: ".
 */
export function decisionSummary(r: FormMemoryResponse): string {
  if (!r.candidates.length) return "nothing worth remembering in that exchange";
  const counts = new Map<MemoryDecision, number>();
  for (const c of r.candidates) counts.set(c.decision, (counts.get(c.decision) ?? 0) + 1);
  return DECISIONS.filter((d) => counts.get(d))
    .map((d) => `${counts.get(d)} ${decisionWord(d)}`)
    .join(", ");
}

function timestamp(ts: { seconds: bigint; nanos: number } | undefined, empty: string): string {
  if (!ts || (!ts.seconds && !ts.nanos)) return empty;
  return new Date(Number(ts.seconds) * 1000).toISOString().replace(/\.\d+Z$/, "Z");
}

/**
 * What the formation decided.
 *
 * A formation receipt reports DECISIONS, because the caller asked for nothing in
 * particular and the platform chose. The things a caller cannot infer from row
 * counts are printed whether or not --verbose is set: what was retired, what was
 * dropped, what was flattened, and what was masked.
 */
export function formationLines(r: FormMemoryResponse, verbose: boolean): Line[] {
  const out: Line[] = [];
  if (verbose) {
    for (const c of r.candidates) {
      out.push(["dim", `${decisionWord(c.decision).padEnd(8)} ${candidateText(c)}${decisionNote(c)}`]);
    }
    // A REVISED candidate's replacement is counted as a supersession, not in
    // vector or edge rows, so a turn that only corrected things legitimately
    // reports zero rows. Printing both side by side is the difference between
    // "nothing was stored" and "what was stored replaced something".
    out.push([
      "dim",
      `formed: log=${r.executionLogRows} vec=${r.vectorRows}(+${r.chunkSupersessions} superseded) ` +
        `nodes=${r.graphNodeRows} edges=${r.graphEdgeRows}(+${r.edgeSupersessions} superseded) ` +
        `@ ${timestamp(r.commitTimestamp, "nothing committed")}`,
    ]);
  } else {
    out.push(["dim", `[formed: ${decisionSummary(r)}]`]);
  }

  // Retired is not deleted: the old assertion keeps its validity window and
  // stays readable as history. Printed always, because it is the one outcome
  // that changes what the workspace says it knows.
  const n = r.edgeSupersessions + r.chunkSupersessions;
  if (n > 0n) {
    out.push([
      "cyan",
      `[memory] ${n} earlier assertion(s) retired by a correction in this turn ` +
        "(superseded, not overwritten: the previous value stays readable as history)",
    ]);
  }
  if (r.candidatesDropped > 0) {
    out.push([
      "yellow",
      `[memory] ${r.candidatesDropped} candidate(s) past the per-formation cap of ` +
        `${r.candidateCap} were dropped, so not everything in that exchange was considered.`,
    ]);
  }
  for (const s of r.summarizedStructures) {
    out.push([
      "yellow",
      `[memory] turn ${s.turnIndex}: ${s.description} ` +
        `(${s.summarizedCount} item(s) summarized rather than stored individually)`,
    ]);
  }
  for (const rd of r.redactions) {
    out.push([
      "yellow",
      `[memory] turn ${rd.turnIndex}: ${rd.maskedCount} value(s) masked before the extraction model saw them`,
    ]);
  }
  return out;
}

/**
 * What an authored commit wrote.
 *
 * A truncated chunk is printed whether or not --verbose is set: the commit
 * succeeded, but the embedding covers only the start of the text, so recall can
 * no longer find the turn by anything said in the part that was cut. This demo
 * does not set rejectOnTruncation, because for a chatbot losing the turn is
 * worse than remembering most of it.
 */
export function commitLines(r: CommitMemoryResponse, verbose: boolean): Line[] {
  const out: Line[] = [];
  if (verbose) {
    out.push([
      "dim",
      `committed: log=${r.executionLogRows} vec=${r.vectorRows} nodes=${r.graphNodeRows} ` +
        `edges=${r.graphEdgeRows} @ ${timestamp(r.commitTimestamp, "?")}`,
    ]);
  }
  if (r.truncatedChunkIds.length) {
    out.push([
      "yellow",
      `[memory] that message was too long to embed in full (${r.truncatedChunkIds.join(", ")}). ` +
        "It is stored, but recall may miss the end of it.",
    ]);
  }
  return out;
}
