/* ══════════════════════════════════════════════════════════════════════════════
   NLLB Translator  ·  script.js
   Handles: translate, pivot UI, Chrome Speech API, RLHF option display,
            params panel, copy/speak, swap languages, character count.
══════════════════════════════════════════════════════════════════════════════ */

"use strict";

// ─── BCP-47 map for SpeechSynthesis / SpeechRecognition ──────────────────────
const BCP47 = {
    vi: "vi-VN", en: "en-US", fil: "fil-PH",
    th: "th-TH", id: "id-ID", ms: "ms-MY",
    khm: "km-KH", lo: "lo-LA", my: "my-MM",
};

// ─── DOM refs ────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const srcLang = $("srcLang");
const tgtLang = $("tgtLang");
const pivotToggle = $("pivotToggle");
const pivotLang = $("pivotLang");
const srcText = $("srcText");
const tgtText = $("tgtText");
const charCount = $("charCount");
const statusArea = $("statusArea");
const translateBtn = $("translateBtn");
const micBtn = $("micBtn");
const speakBtn = $("speakBtn");
const copyBtn = $("copyBtn");
const clearBtn = $("clearBtn");
const swapBtn = $("swapBtn");
const rlhfPanel = $("rlhfPanel");
const rlhfBadge = $("rlhfBadge");
const optionA = $("optionA");
const optionB = $("optionB");
const optionAText = $("optionAText");
const optionBText = $("optionBText");
const toast = $("toast");
// params
const paramMethod = $("paramMethod");
const paramMaxLen = $("paramMaxLen");
const paramNumBeams = $("paramNumBeams");
const paramTemp = $("paramTemp");
const paramTopK = $("paramTopK");
const paramTopP = $("paramTopP");

// ─── State ───────────────────────────────────────────────────────────────────
let lastSrcText = "";
let lastOptionA = "";
let lastOptionB = "";
let translating = false;
let recognition = null;

// ─── Char count ──────────────────────────────────────────────────────────────
srcText.addEventListener("input", () => {
    charCount.textContent = `${srcText.value.length} ký tự`;
});

// ─── Swap languages ──────────────────────────────────────────────────────────
swapBtn.addEventListener("click", () => {
    const tmp = srcLang.value;
    srcLang.value = tgtLang.value;
    tgtLang.value = tmp;
    // swap text too
    const srcVal = srcText.value;
    srcText.value = tgtText.textContent.includes("Bản dịch sẽ") ? "" : tgtText.textContent;
    tgtText.innerHTML = srcVal
        ? srcVal
        : `<span class="placeholder-text">Bản dịch sẽ xuất hiện ở đây…</span>`;
    charCount.textContent = `${srcText.value.length} ký tự`;
});

// ─── Pivot toggle ────────────────────────────────────────────────────────────
pivotToggle.addEventListener("change", () => {
    pivotLang.hidden = !pivotToggle.checked;
});

// ─── Clear button ────────────────────────────────────────────────────────────
clearBtn.addEventListener("click", () => {
    srcText.value = "";
    charCount.textContent = "0 ký tự";
    resetOutput();
});

function resetOutput() {
    tgtText.innerHTML = `<span class="placeholder-text">Bản dịch sẽ xuất hiện ở đây…</span>`;
    speakBtn.disabled = true;
    copyBtn.disabled = true;
    hideRlhf();
}

// ─── Parameters panel binding ─────────────────────────────────────────────────
function bindSlider(slider, display, fmt) {
    slider.addEventListener("input", () => {
        display.textContent = fmt ? fmt(slider.value) : slider.value;
    });
}
bindSlider(paramMaxLen, $("maxLenVal"), null);
bindSlider(paramNumBeams, $("numBeamsVal"), null);
bindSlider(paramTemp, $("tempVal"), v => parseFloat(v).toFixed(2));
bindSlider(paramTopK, $("topKVal"), null);
bindSlider(paramTopP, $("topPVal"), v => parseFloat(v).toFixed(2));

// Show/hide params based on method
paramMethod.addEventListener("change", () => {
    const method = paramMethod.value;
    document.querySelectorAll(".sampling-only").forEach(el => {
        el.style.display = method === "sampling" ? "flex" : "none";
    });
    $("beamGroup").style.display = method === "beam" ? "flex" : "none";
});

function getParams() {
    return {
        method: paramMethod.value,
        max_length: parseInt(paramMaxLen.value),
        num_beams: parseInt(paramNumBeams.value),
        temperature: parseFloat(paramTemp.value),
        top_k: parseInt(paramTopK.value),
        top_p: parseFloat(paramTopP.value),
    };
}

// ─── TRANSLATE ────────────────────────────────────────────────────────────────
translateBtn.addEventListener("click", doTranslate);
srcText.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) doTranslate();
});

async function doTranslate() {
    const text = srcText.value.trim();
    if (!text) { showToast("Hãy nhập văn bản cần dịch"); return; }
    if (translating) return;

    translating = true;
    setLoading(true);
    hideRlhf();
    lastSrcText = text;

    const body = {
        text,
        src: srcLang.value,
        tgt: tgtLang.value,
        use_pivot: pivotToggle.checked,
        pivot: pivotLang.value,
        params: getParams(),
    };

    try {
        const res = await fetchWithTimeout("/translate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        }, 120_000);

        const data = await res.json();

        if (!res.ok) {
            showError(data.error || "Lỗi không xác định");
            return;
        }

        if (data.is_rlhf) {
            lastOptionA = data.options[0];
            lastOptionB = data.options[1];
            showRlhf(data.options[0], data.options[1]);
        } else {
            renderTranslation(data.translated_text);
        }

    } catch (err) {
        showError(err.message === "timeout"
            ? "Quá thời gian chờ — model lớn cần nhiều thời gian hơn. Hãy thử lại."
            : `Lỗi mạng: ${err.message}`);
    } finally {
        translating = false;
        setLoading(false);
    }
}

function setLoading(on) {
    translateBtn.classList.toggle("loading", on);
    translateBtn.querySelector(".btn-text").textContent = on ? "Đang dịch…" : "Dịch";
    statusArea.innerHTML = on
        ? `<span class="spinner"></span> Đang xử lý…`
        : "";
}

function renderTranslation(text) {
    tgtText.textContent = text;
    speakBtn.disabled = false;
    copyBtn.disabled = false;
    rlhfBadge.hidden = true;
    statusArea.innerHTML = "✓ Hoàn thành";
    setTimeout(() => { statusArea.innerHTML = ""; }, 3000);
}

function showError(msg) {
    tgtText.innerHTML = `<span style="color:var(--red)">${msg}</span>`;
    statusArea.innerHTML = "";
}

// ─── RLHF ────────────────────────────────────────────────────────────────────
function showRlhf(a, b) {
    optionAText.textContent = a;
    optionBText.textContent = b;
    rlhfPanel.hidden = false;
    rlhfBadge.hidden = false;
    // Clear previous state
    optionA.className = "rlhf-option";
    optionB.className = "rlhf-option";
    statusArea.innerHTML = "⊕ Chọn bản dịch tốt hơn";
}

function hideRlhf() {
    rlhfPanel.hidden = true;
    rlhfBadge.hidden = true;
}

async function submitRlhf(chosen, rejected) {
    optionA.disabled = true;
    optionB.disabled = true;
    renderTranslation(chosen);

    try {
        await fetch("/rlhf_feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ source: lastSrcText, chosen, rejected }),
        });
        showToast("✓ Phản hồi đã được lưu — cảm ơn!");
    } catch {
        showToast("Không thể lưu phản hồi (lỗi mạng)");
    } finally {
        optionA.disabled = false;
        optionB.disabled = false;
    }
}

optionA.addEventListener("click", () => {
    optionA.classList.add("chosen");
    optionB.classList.add("rejected");
    submitRlhf(lastOptionA, lastOptionB);
});
optionB.addEventListener("click", () => {
    optionB.classList.add("chosen");
    optionA.classList.add("rejected");
    submitRlhf(lastOptionB, lastOptionA);
});

// ─── Copy ────────────────────────────────────────────────────────────────────
copyBtn.addEventListener("click", async () => {
    const text = tgtText.textContent;
    if (!text) return;
    try {
        await navigator.clipboard.writeText(text);
        showToast("Đã sao chép vào clipboard");
    } catch {
        showToast("Không thể sao chép — hãy copy thủ công");
    }
});

// ─── Text-to-Speech ──────────────────────────────────────────────────────────
speakBtn.addEventListener("click", () => {
    const text = tgtText.textContent;
    if (!text || window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
        return;
    }
    const utt = new SpeechSynthesisUtterance(text);
    utt.lang = BCP47[tgtLang.value] || "en-US";
    utt.rate = 0.95;
    window.speechSynthesis.speak(utt);
    showToast(`🔊 Đọc bằng giọng ${utt.lang}`);
});

// ─── Speech-to-Text (Web Speech API) ────────────────────────────────────────
micBtn.addEventListener("click", () => {
    if (!("webkitSpeechRecognition" in window || "SpeechRecognition" in window)) {
        showToast("Trình duyệt không hỗ trợ Web Speech API — hãy dùng Chrome");
        return;
    }

    if (recognition) {
        recognition.stop();
        recognition = null;
        micBtn.classList.remove("recording");
        return;
    }

    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.lang = BCP47[srcLang.value] || "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;

    micBtn.classList.add("recording");

    recognition.onresult = e => {
        let interim = "", final = "";
        for (const result of e.results) {
            if (result.isFinal) final += result[0].transcript;
            else interim += result[0].transcript;
        }
        srcText.value = (srcText.value + " " + final).trim() || interim;
        charCount.textContent = `${srcText.value.length} ký tự`;
    };

    recognition.onerror = err => {
        showToast(`Lỗi micro: ${err.error}`);
        micBtn.classList.remove("recording");
        recognition = null;
    };

    recognition.onend = () => {
        micBtn.classList.remove("recording");
        recognition = null;
    };

    recognition.start();
    showToast("🎙 Đang lắng nghe…");
});

// ─── Toast helper ─────────────────────────────────────────────────────────────
let toastTimer;
function showToast(msg, duration = 3000) {
    toast.textContent = msg;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), duration);
}

// ─── Fetch with timeout ───────────────────────────────────────────────────────
function fetchWithTimeout(url, options, ms) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), ms);
    return fetch(url, { ...options, signal: controller.signal })
        .then(r => { clearTimeout(timer); return r; })
        .catch(err => {
            clearTimeout(timer);
            if (err.name === "AbortError") throw new Error("timeout");
            throw err;
        });
}