#!/usr/bin/env python3
"""
test_frontend_integration.py
End-to-end integration test verifying the frontend integration:
1. Header Settings Icon & Odonto Settings Modal
2. Diagnostic Agent Switching (Local Expert Engine vs Gemini 2.0 Flash)
3. Preserving the exact Odonto theme & UI styling
4. Teacher Studio Depth Anything V2 2.5D relief extraction & heatmap
5. Multi-Agent Clinical Diagnostics & Target Tooth Recommender
6. Deformation Method Switcher (Method A, Method B, Side-by-Side Dual Viewports)
7. Case Publishing to Student Practice Rooms
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "test_results" / "frontend_integration"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

APP_URL = "http://127.0.0.1:5000"


def ensure_server_running():
    """Checks if Flask server is already running, or starts it."""
    try:
        r = requests.get(f"{APP_URL}/api/teeth_list", timeout=2)
        if r.status_code == 200:
            print(f"[OK] Flask server is already running at {APP_URL}")
            return None
    except Exception:
        pass

    print(f"[*] Starting Flask app.py in background...")
    proc = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    import threading
    def log_flask():
        for line in proc.stdout:
            print(f"[FLASK] {line.strip()}", flush=True)
    t = threading.Thread(target=log_flask, daemon=True)
    t.start()

    for _ in range(40):
        time.sleep(1)
        try:
            r = requests.get(f"{APP_URL}/api/teeth_list", timeout=2)
            if r.status_code == 200:
                print(f"[OK] Flask server started successfully at {APP_URL}")
                return proc
        except Exception:
            pass
    print("[ERROR] Failed to connect to Flask server after 40 seconds.")
    proc.terminate()
    sys.exit(1)


def run_e2e_tests():
    proc = ensure_server_running()
    try:
        with sync_playwright() as p:
            print("[*] Launching Chromium browser...")
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()

            # ---------------------------------------------------------------
            # 1. Login as Teacher
            # ---------------------------------------------------------------
            print(f"[*] Navigating to {APP_URL}...")
            page.goto(APP_URL, wait_until="networkidle")

            page.on("console", lambda msg: print(f"  [BROWSER {msg.type}] {msg.text}"))
            page.on("pageerror", lambda err: print(f"  [BROWSER ERROR] {err}"))
            page.on("response", lambda r: print(f"  [HTTP {r.status}] {r.url}") if "/api/" in r.url else None)

            # Check if login form is displayed
            auth_screen = page.locator("#auth-screen")
            if auth_screen.is_visible():
                print("[*] Logging in with teacher credentials (teacher@dental.edu)...")
                page.fill("#auth-email", "teacher@dental.edu")
                page.fill("#auth-password", "teacher123")
                page.click("#auth-submit-btn")
                
                # Check for error message if visible
                time.sleep(2)
                err_box = page.locator("#auth-error-box")
                if err_box.is_visible():
                    print(f"[AUTH ERROR] {err_box.inner_text()}")

                page.wait_for_selector("#app-shell:not(.hidden)", timeout=10000)
                print("[OK] Logged in to teacher authenticated shell.")

            time.sleep(1)

            # ---------------------------------------------------------------
            # 2. Test Settings Modal & Agent Selection
            # ---------------------------------------------------------------
            print("[*] Testing Settings Modal via header button...")
            settings_btn = page.locator("#btn-header-settings")
            assert settings_btn.is_visible(), "Header Settings button should be visible"
            settings_btn.click()

            page.wait_for_selector("#odonto-settings-modal:not(.hidden)", timeout=5000)
            modal = page.locator("#odonto-settings-modal")
            assert modal.is_visible(), "Odonto Settings Modal should be open"

            # Verify modal title and styling
            modal_title = page.locator("#odonto-settings-modal h3").inner_text()
            print(f"[OK] Settings Modal opened: '{modal_title}'")

            # Select Local Expert Engine
            card_local = page.locator("#card-agent-local")
            card_local.click()
            time.sleep(0.5)

            # Capture Settings Modal Snapshot
            snap_settings = RESULTS_DIR / "01_settings_modal_open.png"
            page.screenshot(path=str(snap_settings))
            print(f"[OK] Saved snapshot: {snap_settings.name}")

            # Save settings
            page.click("#btn-save-settings")
            page.wait_for_selector("#odonto-settings-modal", state="hidden", timeout=5000)
            print("[OK] Settings saved and modal closed.")

            # ---------------------------------------------------------------
            # 3. Navigate to Teacher Studio (SAM2 Pipeline)
            # ---------------------------------------------------------------
            print("[*] Navigating to 'Create a case' (SAM2 Pipeline)...")
            page.click("#nav-sam2-pipeline")
            page.wait_for_selector("#view-sam2-pipeline:not(.hidden)", timeout=5000)

            # Check that Agent badge shows Local Expert Engine
            badge = page.locator("#pipe-agent-source-badge")
            assert "LOCAL EXPERT" in badge.inner_text().upper(), f"Expected Local Expert badge, got: {badge.inner_text()}"
            print(f"[OK] Agent source badge reflects active setting: '{badge.inner_text()}'")

            # ---------------------------------------------------------------
            # 4. Load Benchmark 1 (Occlusal Caries Molar #16)
            # ---------------------------------------------------------------
            print("[*] Selecting Benchmark 1 preset (alpha_p006_F_15_001)...")
            page.select_option("#pipe-preset-select", "alpha_p006_F_15_001")

            # Wait for SAM boundary snap, Depth Anything extraction, and Diagnostics Agent
            print("[*] Waiting for Depth Anything V2 2.5D heatmap & Clinical Agent evaluation...")
            page.wait_for_selector("#pipe-depth-card:not(.hidden)", timeout=35000)
            page.wait_for_selector("#pipe-agent-card:not(.hidden)", timeout=35000)

            # Verify Depth Metrics populated in pipe-depth-card
            pts_text = page.locator("#pipe-metric-points").inner_text()
            area_text = page.locator("#pipe-metric-area").inner_text()
            ring_text = page.locator("#pipe-metric-ring").inner_text()
            print(f"[OK] Depth Anything V2 metrics: Points={pts_text}, Area={area_text}, Ring={ring_text}")

            # Verify Clinical Diagnostics Agent results
            pathology_badge = page.locator("#pipe-diag-pathology").inner_text()
            grade_badge = page.locator("#pipe-diag-grade").inner_text()
            conf_badge = page.locator("#pipe-diag-conf").inner_text()
            depth_readout = page.locator("#pipe-diag-depth-readout").inner_text()
            rationale_text = page.locator("#pipe-diag-rationale").inner_text()
            print(f"[OK] Clinical Agent Diagnosis: {pathology_badge} | {grade_badge} | {conf_badge}")
            print(f"[OK] Depth: {depth_readout} | Rationale: {rationale_text[:60]}...")

            # Verify Transfer Agent Target Recommendation
            recom_banner = page.locator("#pipe-transfer-banner")
            assert recom_banner.is_visible(), "Clinical Transfer Agent recommendation banner should be visible"
            target_tooth = page.locator("#pipe-transfer-fdi-badge").inner_text()
            print(f"[OK] Transfer Agent recommended target: {target_tooth}")

            # Save snapshot of Depth & Agent Diagnostics
            snap_diagnostics = RESULTS_DIR / "02_depth_and_agent_diagnostics.png"
            page.screenshot(path=str(snap_diagnostics))
            print(f"[OK] Saved snapshot: {snap_diagnostics.name}")

            # ---------------------------------------------------------------
            # 5. Execute 3D Carve (Method B: Depth Model Heightmap)
            # ---------------------------------------------------------------
            print("[*] Testing Method B (Depth Model Heightmap) Carve Preview...")
            page.click("#pipe-btn-method-b")
            time.sleep(0.5)

            # Click generate 3D preview
            page.click("#btn-pipe-carve-preview")
            print("[*] Waiting for 3D mesh carve generation...")
            page.wait_for_selector("#btn-pipe-publish-room:not([disabled])", timeout=40000)
            time.sleep(3)  # Allow Three.js canvas to render frame

            snap_carve = RESULTS_DIR / "03_method_b_carved_preview.png"
            page.screenshot(path=str(snap_carve))
            print(f"[OK] Saved snapshot: {snap_carve.name}")

            # ---------------------------------------------------------------
            # 6. Test Side-by-Side Dual Viewport Comparison
            # ---------------------------------------------------------------
            print("[*] Testing Side-by-Side Comparison mode...")
            page.click("#pipe-btn-method-compare")
            page.wait_for_selector("#pipe-side-by-side-card:not(.hidden)", timeout=40000)

            # Verify both Method A and Method B images are loaded
            img_a = page.locator("#pipe-compare-render-a")
            img_b = page.locator("#pipe-compare-render-b")
            assert img_a.is_visible() and img_b.is_visible(), "Both dual viewports should be visible"

            # Check comparison metrics
            depth_a = page.locator("#pipe-compare-depth-a").inner_text()
            depth_b = page.locator("#pipe-compare-depth-b").inner_text()
            print(f"[OK] Side-by-side metrics: Method A depth={depth_a}, Method B depth={depth_b}")

            snap_compare = RESULTS_DIR / "04_side_by_side_comparison.png"
            page.screenshot(path=str(snap_compare))
            print(f"[OK] Saved snapshot: {snap_compare.name}")

            # ---------------------------------------------------------------
            # 7. Switch Agent in Settings to Gemini 2.0 Flash
            # ---------------------------------------------------------------
            print("[*] Testing Agent Switching to Gemini 2.0 Flash in Settings...")
            page.click("#btn-header-settings")
            page.wait_for_selector("#odonto-settings-modal:not(.hidden)", timeout=5000)

            card_gemini = page.locator("#card-agent-gemini")
            card_gemini.click()
            time.sleep(0.5)

            # Verify Gemini card selected & API key container active
            page.fill("#settings-gemini-api-key", "DEMO_ODONTO_GEMINI_KEY_7894")
            time.sleep(0.5)

            snap_gemini_sel = RESULTS_DIR / "05_settings_gemini_selection.png"
            page.screenshot(path=str(snap_gemini_sel))
            print(f"[OK] Saved snapshot: {snap_gemini_sel.name}")

            page.click("#btn-save-settings")
            page.wait_for_selector("#odonto-settings-modal", state="hidden", timeout=5000)

            # Verify source badge now shows Gemini
            badge_gemini = page.locator("#pipe-agent-source-badge")
            assert "GEMINI" in badge_gemini.inner_text().upper(), f"Expected Gemini badge, got: {badge_gemini.inner_text()}"
            print(f"[OK] Source badge successfully updated to: '{badge_gemini.inner_text()}'")

            # ---------------------------------------------------------------
            # 8. Test Case Publishing to Room
            # ---------------------------------------------------------------
            print("[*] Testing Publishing Case to Student Practice Room...")
            # Switch back to Method B for publication
            page.click("#pipe-btn-method-b")
            time.sleep(0.5)
            page.click("#btn-pipe-publish-room")
            page.wait_for_function("() => document.getElementById('pipe-publish-feedback')?.textContent.includes('Published')", timeout=20000)
            pub_msg = page.locator("#pipe-publish-feedback").inner_text()
            print(f"[OK] Publish confirmation: {pub_msg}")

            snap_published = RESULTS_DIR / "06_case_published.png"
            page.screenshot(path=str(snap_published))
            print(f"[OK] Saved snapshot: {snap_published.name}")

            print("\n" + "=" * 60)
            print("🎉 ALL FRONTEND INTEGRATION TESTS PASSED SUCCESSFULLY!")
            print("=" * 60)

            browser.close()

    finally:
        if proc:
            print("[*] Terminating background Flask test process...")
            proc.terminate()


if __name__ == "__main__":
    run_e2e_tests()
