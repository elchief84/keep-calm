/* Content script: detects message fields, analyzes them on input, shows a badge. */

const INTENT_DESC = {
  constructive: "costruttivo",
  critical: "critico",
  personal: "personale (attacca la persona)",
  informational: "informativo",
};

const TONE_DESC = {
  neutral: "neutro",
  frustrated: "frustrato",
  hostile: "ostile",
  sarcastic: "sarcastico",
  positive: "positivo",
};

const LEVEL_DESC = {
  none: "Nessun rischio",
  low: "Rischio basso",
  medium: "Rischio medio",
  high: "Rischio alto",
  critical: "Rischio critico",
};

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function activeText(element) {
  if (element.tagName === "TEXTAREA" || element.tagName === "INPUT") {
    return element.value;
  }
  if (element.isContentEditable) {
    return element.innerText;
  }
  return "";
}

function badgeColor(level) {
  return level === "none" || level === "low" ? "#16a34a"
    : level === "medium" ? "#d97706"
    : "#dc2626";
}

function showPopup(badge, result) {
  closePopup();

  const popup = document.createElement("div");
  popup.className = "keepcalm-popup";
  popup.style.cssText =
    "position:fixed; z-index:100000; font:13px/1.5 system-ui; padding:12px 14px;" +
    "border-radius:8px; background:#fff; color:#111; box-shadow:0 4px 16px rgba(0,0,0,.25);" +
    "max-width:320px;";

  const tones = result.tones.map((t) => TONE_DESC[t.label] || t.label).join(", ");
  const intent = INTENT_DESC[result.intent] || result.intent;

  popup.innerHTML = `
    <div style="font-weight:700; margin-bottom:6px;">
      ${LEVEL_DESC[result.risk_level]} — ${Math.round(result.risk * 100)}%
    </div>
    <div style="margin-bottom:4px;"><strong>Intento:</strong> ${intent}</div>
    ${tones ? `<div style="margin-bottom:4px;"><strong>Toni:</strong> ${tones}</div>` : ""}
    <div style="color:#444;">${result.explanation}</div>
  `;

  const rect = badge.getBoundingClientRect();
  popup.style.left = Math.max(8, rect.right - popup.offsetWidth) + "px";
  popup.style.top = rect.bottom + 6 + "px";

  document.body.appendChild(popup);
  badge.__keepCalmPopup = popup;
}

function closePopup() {
  document.querySelectorAll(".keepcalm-popup").forEach((p) => p.remove());
}

function showBadge(element, result) {
  let badge = element.__keepCalmBadge;
  if (!badge) {
    badge = document.createElement("div");
    badge.style.cssText =
      "position:fixed; z-index:99999; font:12px system-ui; padding:4px 8px;" +
      "border-radius:6px; color:#fff; cursor:pointer;";
    document.body.appendChild(badge);
    element.__keepCalmBadge = badge;
    badge.addEventListener("click", (e) => {
      e.stopPropagation();
      showPopup(badge, element.__keepCalmResult);
    });
  }

  element.__keepCalmResult = result;
  badge.style.background = badgeColor(result.risk_level);
  badge.textContent = `🧘 ${Math.round(result.risk * 100)}% · ${INTENT_DESC[result.intent] || result.intent}`;

  const rect = element.getBoundingClientRect();
  badge.style.left = rect.right - badge.offsetWidth + "px";
  badge.style.top = rect.top - badge.offsetHeight - 4 + "px";
}

function hideBadge(element) {
  if (element.__keepCalmBadge) {
    element.__keepCalmBadge.remove();
    element.__keepCalmBadge = null;
  }
  if (element.__keepCalmPopup) {
    element.__keepCalmPopup.remove();
    element.__keepCalmPopup = null;
  }
}

document.addEventListener("click", closePopup, true);

const analyze = debounce(async (element) => {
  const text = activeText(element);
  if (!text || text.length < 2) {
    hideBadge(element);
    return;
  }
  try {
    const result = await chrome.runtime.sendMessage({ type: "ANALYZE", text });
    if (result && !result.error) {
      showBadge(element, result);
    } else {
      hideBadge(element);
    }
  } catch {
    hideBadge(element);
  }
}, 600);

function attach(element) {
  if (element.__keepCalmAttached) return;
  element.__keepCalmAttached = true;
  element.addEventListener("input", () => analyze(element));
  element.addEventListener("blur", () => hideBadge(element));
}

function scan() {
  document.querySelectorAll("textarea, input[type='text'], [contenteditable='true']")
    .forEach(attach);
}

const observer = new MutationObserver(scan);
observer.observe(document.body, { childList: true, subtree: true });
scan();
