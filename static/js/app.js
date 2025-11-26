// static/js/app.js  -- Drop this in and overwrite previous file
// Version 1.0 unified handler for Tivaan Vision UI -> Backend

console.log("[TivaanVision app.js] loaded", new Date().toISOString());

(function () {
  "use strict";

  // small helper to create element
  function $id(id) { return document.getElementById(id); }
  function qs(sel, root=document) { return root.querySelector(sel); }
  function qsa(sel, root=document) { return Array.from(root.querySelectorAll(sel)); }

  // --- theme toggle (site-wide) ---
  function injectThemeToggle() {
    // add control if missing
    if (document.body.querySelector("#tv-theme-toggle")) return;
    const btn = document.createElement("button");
    btn.id = "tv-theme-toggle";
    btn.title = "Toggle theme";
    btn.style.position = "fixed";
    btn.style.top = "12px";
    btn.style.right = "12px";
    btn.style.zIndex = "9999";
    btn.style.padding = "8px 10px";
    btn.style.borderRadius = "8px";
    btn.style.border = "none";
    btn.style.cursor = "pointer";
    btn.style.background = "rgba(0,0,0,0.35)";
    btn.style.color = "#fff";
    btn.textContent = document.body.classList.contains("light") ? "☀️" : "🌙";
    btn.addEventListener("click", () => {
      document.body.classList.toggle("light");
      btn.textContent = document.body.classList.contains("light") ? "☀️" : "🌙";
    });
    document.body.appendChild(btn);
  }

  // call immediately
  injectThemeToggle();

  // --- utility UI helpers ---
  function setStatus(el, html, klass) {
    if (!el) return;
    el.innerHTML = html;
    el.className = klass ? klass : "";
  }
  function showImg(imgEl, src) {
    if (!imgEl) return;
    if (!src) { imgEl.style.display = "none"; return; }
    imgEl.src = src;
    imgEl.style.display = "block";
  }

  // --- IoT simulation call (POST) ---
  async function callIot(distanceCm) {
    try {
      const res = await fetch("/api/iot", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ distance: distanceCm })
      });
      if (!res.ok) {
        console.warn("[iot] non-ok", res.status);
        return null;
      }
      return await res.json();
    } catch (e) {
      console.error("[iot] error", e);
      return null;
    }
  }

  // --- auto-update metrics images (cache-busted) ---
  function refreshMetricsThumbnails() {
    const imgs = qsa(".metrics-thumb, img.metrics-thumb");
    const t = Date.now();
    imgs.forEach(img => {
      const base = img.getAttribute("src").split("?")[0];
      img.src = base + "?t=" + t;
    });
  }

  // --- core: find all run-detect buttons and attach behavior ---
  function attachDetectHandlers() {
    // selector: any element with class 'run-detect' (button or input)
    const runButtons = qsa(".run-detect");
    if (!runButtons.length) {
      console.warn("[Tivaan] No .run-detect buttons found on page");
    }
    runButtons.forEach(btn => {
      // avoid double attaching
      if (btn.dataset.tvAttached === "1") return;
      btn.dataset.tvAttached = "1";

      // locate nearest file input (look in same form or parent containers)
      const container = btn.closest("form") || btn.closest(".card") || document.body;
      const fileInput = container.querySelector('input[type="file"]') || container.querySelector('input[name="file"]') || container.querySelector('input[name="image"]');

      // locate UI elements for status, preview, annotated, iot
      const statusEl = container.querySelector(".tv-status") || container.querySelector("#status") || (() => {
        const d = document.createElement("div");
        d.className = "tv-status";
        btn.insertAdjacentElement("afterend", d);
        return d;
      })();

      const previewEl = container.querySelector(".tv-preview") || container.querySelector("#preview");
      const annotatedEl = container.querySelector(".tv-annotated") || container.querySelector("#resultImg");
      const metaEl = container.querySelector(".tv-meta") || container.querySelector("#meta");
      const iotEl = container.querySelector(".tv-iot") || container.querySelector("#iotOutput");

      // click handler
      btn.addEventListener("click", async (ev) => {
        ev.preventDefault();
        setStatus(statusEl, "<em>Processing…</em>", "processing");

        // check file input
        if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
          setStatus(statusEl, "<span style='color:orange'>Please choose an image first</span>", "warning");
          return;
        }

        // show preview
        if (previewEl) {
          const f = fileInput.files[0];
          try {
            previewEl.src = URL.createObjectURL(f);
            previewEl.style.display = "block";
          } catch (e) {
            console.warn("preview fail", e);
          }
        }

        // disable button while processing
        btn.disabled = true;

        // prepare formdata with key 'file' (backend expects this)
        const fd = new FormData();
        fd.append("file", fileInput.files[0]);

        // send to backend
        let json = null;
        try {
          const resp = await fetch("/api/detect", { method: "POST", body: fd });
          if (!resp.ok) {
            const txt = await resp.text().catch(()=>"");
            setStatus(statusEl, `<span style='color:red'>Server error ${resp.status}</span>`, "error");
            console.error("[detect] server error", resp.status, txt);
            btn.disabled = false;
            return;
          }

          // try parse JSON text
          const txt = await resp.text();
          try { json = JSON.parse(txt); }
          catch (e) {
            console.error("[detect] Response not JSON:", e, txt.slice(0,400));
            setStatus(statusEl, "<span style='color:red'>Invalid server response — check backend logs</span>", "error");
            btn.disabled = false;
            return;
          }
        } catch (e) {
          console.error("[detect] fetch error", e);
          setStatus(statusEl, "<span style='color:red'>Network error — backend may be down</span>", "error");
          btn.disabled = false;
          return;
        }

        // expected fields: output_image (string path or data URL), vehicle_count (int)
        console.debug("[detect] json:", json);
        const outImg = json.output_image || json.annotated || json.image || null;
        const count = json.vehicle_count ?? json.count ?? json.vehicles ?? null;
        const risk = json.risk_level || json.risk || null;

        // update UI
        setStatus(statusEl, "<b>Detection completed</b>", "ok");
        if (metaEl) metaEl.innerHTML = `<b>Vehicles:</b> ${count ?? "N/A"} &nbsp;&nbsp; <b>Risk:</b> ${risk ?? "N/A"}`;

        if (annotatedEl) {
          if (outImg) {
            // if backend returned data URL, use it directly; else use path
            annotatedEl.src = outImg;
            annotatedEl.style.display = "block";
          } else {
            annotatedEl.style.display = "none";
          }
        }

        // IoT auto-run using simple heuristic
        if (count !== null && iotEl) {
          const distance = Math.max(5, Math.round(140 - count*2.0));
          const iotResp = await callIot(distance);
          if (iotResp) {
            iotEl.innerHTML = `<b>Distance:</b> ${iotResp.distance} cm &nbsp; <b>Alert:</b> ${iotResp.alert} &nbsp; <b>Action:</b> ${iotResp.recommended_action}`;
          } else {
            iotEl.innerHTML = "IoT sim failed — see console.";
          }
        }

        // refresh metrics thumbnails on site to pick up latest images
        try { refreshMetricsThumbnails(); } catch (e) {}

        btn.disabled = false;
      });
    });
  }

  // run initial attach
  attachDetectHandlers();

  // expose functions to console for debugging
  window.TivaanVision = {
    attachDetectHandlers,
    injectThemeToggle,
    refreshMetricsThumbnails
  };

  console.log("[TivaanVision] handlers attached. If you add new DOM elements, call window.TivaanVision.attachDetectHandlers()");
})();
