/* Background service worker: routes analysis requests to the offscreen
 * document that hosts the ONNX model. */

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  await chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["WORKERS"],
    justification: "Run the Keep Calm ONNX model for local message analysis",
  });
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "ANALYZE") {
    analyze(message.text)
      .then(sendResponse)
      .catch((err) => sendResponse({ error: err.message }));
    return true;
  }
});

async function analyze(text) {
  await ensureOffscreen();
  return chrome.runtime.sendMessage({ type: "ANALYZE_OFFSCREEN", text });
}
