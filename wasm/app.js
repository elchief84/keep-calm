/* Keep Calm in-browser inference using ONNX Runtime Web. */

import { loadTokenizer } from "./tokenizer.js";

const TONE_LABELS = ["neutral", "frustrated", "hostile", "sarcastic", "positive"];
const INTENT_LABELS = ["constructive", "critical", "personal", "informational"];
const RISK_LEVELS = [
  [0.25, "none"], [0.45, "low"], [0.65, "medium"], [0.85, "high"], [1.0, "critical"],
];

const MODEL_URL = "keep_calm_int8.onnx";

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

async function main() {
  const status = document.getElementById("status");
  const analyzeBtn = document.getElementById("analyze");
  const input = document.getElementById("input");
  const output = document.getElementById("output");

  status.textContent = "Loading tokenizer ...";
  const tokenizer = await loadTokenizer("vocab.json");

  status.textContent = "Loading model (~135MB) ...";
  const ort = window.ort;
  const session = await ort.InferenceSession.create(MODEL_URL);

  status.textContent = "Ready.";
  analyzeBtn.disabled = false;

  analyzeBtn.addEventListener("click", async () => {
    const text = input.value.trim();
    if (!text) return;

    const enc = tokenizer.encode(text, 256);
    const { input_ids, attention_mask } = enc;

    const feedIds = new BigInt64Array(input_ids.map((x) => BigInt(x)));
    const feedMask = new BigInt64Array(attention_mask.map((x) => BigInt(x)));

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

    const level = riskToLevel(risk);
    const needsAttention = risk >= 0.5;
    const explanation = buildExplanation(risk, tones, intent);

    output.innerHTML = `
      <p><strong>Risk:</strong> ${(risk * 100).toFixed(0)}% (${level})</p>
      <p><strong>Intent:</strong> ${intent}</p>
      <p><strong>Tones:</strong> ${
        tones.length ? tones.map((t) => `${t.label} (${(t.confidence * 100).toFixed(0)}%)`).join(", ") : "none"
      }</p>
      <p><strong>Needs attention:</strong> ${needsAttention ? "yes ⚠️" : "no ✅"}</p>
      <p>${explanation}</p>
    `;
  });
}

main().catch((err) => {
  document.getElementById("status").textContent = "Error: " + err.message;
  console.error(err);
});
