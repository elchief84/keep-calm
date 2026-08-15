/* Offscreen document: hosts the ONNX model and tokenizer, runs inference. */

import * as ort from "./ort.min.mjs";
import { loadTokenizer } from "./tokenizer.js";

const TONE_LABELS = ["neutral", "frustrated", "hostile", "sarcastic", "positive"];
const INTENT_LABELS = ["constructive", "critical", "personal", "informational"];
const RISK_LEVELS = [
  [0.25, "none"], [0.45, "low"], [0.65, "medium"], [0.85, "high"], [1.0, "critical"],
];

// Point ONNX Runtime at the local .wasm binaries bundled with the extension.
ort.env.wasm.wasmPaths = chrome.runtime.getURL("");

let session = null;
let tokenizer = null;
let loadPromise = null;

function riskToLevel(score) {
  for (const [threshold, level] of RISK_LEVELS) {
    if (score < threshold) return level;
  }
  return "critical";
}

function buildExplanation(risk, tones, intent) {
  const labels = tones.map((t) => t.label);
  if (labels.includes("hostile") && labels.includes("sarcastic")) {
    return "This message combines sarcasm with hostility — it may be perceived as a personal attack disguised as humor.";
  }
  if (labels.includes("hostile")) {
    return "This message may be perceived as hostile — it appears to target a person rather than addressing an issue.";
  }
  if (labels.includes("sarcastic")) {
    return "This message may read as sarcastic — the tone could be interpreted differently than intended.";
  }
  if (risk < 0.25) return "This message reads as clear and constructive.";
  if (risk < 0.45) return "This message is direct but appears respectful.";
  if (intent === "personal") {
    return "This message appears to target a person rather than the issue.";
  }
  return "This message may be perceived as more critical than intended.";
}

async function ensureLoaded() {
  if (session && tokenizer) return;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    tokenizer = await loadTokenizer(chrome.runtime.getURL("vocab.json"));
    session = await ort.InferenceSession.create(
      chrome.runtime.getURL("model/keep_calm_int8.onnx"),
    );
  })();

  await loadPromise;
}

async function analyze(text) {
  await ensureLoaded();

  const enc = tokenizer.encode(text, 256);
  const feedIds = new BigInt64Array(enc.input_ids.map((x) => BigInt(x)));
  const feedMask = new BigInt64Array(enc.attention_mask.map((x) => BigInt(x)));

  const feeds = {
    input_ids: new ort.Tensor("int64", feedIds, [1, 256]),
    attention_mask: new ort.Tensor("int64", feedMask, [1, 256]),
  };

  const results = await session.run(feeds);
  const risk = results.risk.data[0];
  const tone = Array.from(results.tone.data);
  const intentLogits = Array.from(results.intent.data);

  const intentIdx = intentLogits.indexOf(Math.max(...intentLogits));
  const intent = INTENT_LABELS[intentIdx];

  const tones = TONE_LABELS
    .map((label, i) => ({ label, confidence: tone[i] }))
    .filter((t) => t.confidence >= 0.4)
    .sort((a, b) => b.confidence - a.confidence);

  return {
    risk,
    risk_level: riskToLevel(risk),
    intent,
    tones,
    needs_attention: risk >= 0.5,
    explanation: buildExplanation(risk, tones, intent),
  };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "ANALYZE_OFFSCREEN") {
    analyze(message.text)
      .then(sendResponse)
      .catch((err) => sendResponse({ error: err.message }));
    return true;
  }
});
