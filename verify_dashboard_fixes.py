import asyncio
import os
from playwright.async_api import async_playwright

SCREENSHOTS_DIR = "/home/aracknab/.gemini/antigravity/brain/f151beab-894b-4b6e-80a0-7b0a70c0d516"

async def run_verification():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/usr/bin/chromium",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        context = await browser.new_context(viewport={"width": 1440, "height": 1100})
        page = await context.new_page()

        print("[TEST] Navigating to http://localhost:8055...")
        await page.goto("http://localhost:8055", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # ------------------------------------------------------------------
        # TEST 1: Verification A (Fix 1: Modal Confirmation for 0 Traced Points)
        # ------------------------------------------------------------------
        print("[TEST 1] Testing preset confirmation modal on 0 traced points...")
        
        # Verify preset is dental_1138 or select it
        await page.select_option("#preset-select", "dental_1138_jpg.rf.fd470daf6db30f3a92dd2ccced1d10d7")
        await page.wait_for_timeout(2500)

        # Select manual crack profile and generic preset
        await page.select_option("#manual-profile-select", "v_groove_path")
        await page.select_option("#crack-preset-select", "occlusal_fissure")

        # Clear any existing points
        await page.click("button:has-text('Clear')")
        pts_text = await page.inner_text("#canvas-pts-count")
        print(f"[TEST 1] Points count after clear: {pts_text}")

        # Click Render Manual 3D Carve
        print("[TEST 1] Clicking 'Render Manual 3D Carve' with 0 points...")
        await page.click("#manual-carve-btn")
        await page.wait_for_timeout(800)

        # Assert confirmation modal is visible
        modal_hidden = await page.locator("#preset-confirm-modal").evaluate("el => el.classList.contains('hidden')")
        print(f"[TEST 1] Is modal hidden? {modal_hidden} (Should be False)")
        assert not modal_hidden, "Confirmation modal failed to trigger on 0 points!"

        # Capture snapshot of the confirmation modal
        modal_snap_path = os.path.join(SCREENSHOTS_DIR, "snapshot_fix1_preset_modal_triggered.png")
        await page.screenshot(path=modal_snap_path)
        print(f"[TEST 1] Captured modal snapshot: {modal_snap_path}")

        # Click Proceed with Preset Path
        print("[TEST 1] Clicking 'Proceed with Preset Path' in modal...")
        await page.click("#modal-proceed-btn")

        # Wait for 3D render to complete
        await page.wait_for_selector("#carve-render-container:not(.hidden)", timeout=20000)
        await page.wait_for_timeout(1500)

        # Verify derivation banner and metrics reflect Demo Preset
        banner_text = await page.inner_text("#carve-derivation-banner")
        source_text = await page.inner_text("#metric-source")
        print(f"[TEST 1] Banner text: {banner_text}")
        print(f"[TEST 1] Metric source: {source_text}")
        assert "Demo Preset Carve" in banner_text, "Banner did not indicate Demo Preset!"
        assert "DEMO PRESET" in banner_text, "Banner badge missing DEMO PRESET!"
        assert "Demo Preset" in source_text, "Metric source tile missing Demo Preset!"

        preset_snap_path = os.path.join(SCREENSHOTS_DIR, "snapshot_fix1_preset_render_labeled.png")
        await page.screenshot(path=preset_snap_path)
        print(f"[TEST 1] Captured preset render snapshot: {preset_snap_path}")

        # ------------------------------------------------------------------
        # TEST 2: Verification B (Fix 2: Tooth Match Self-Check & Real Traced Crack Carve)
        # ------------------------------------------------------------------
        print("\n[TEST 2] Testing tooth match sanity check and photo-derived crack trace on Tooth #16...")

        # Select FDI #16 (Upper Right 1st Molar) to match maxillary anatomy of dental_1138
        await page.select_option("#fdi-select", "16")
        await page.wait_for_timeout(500)

        # Verify anatomical self-check badge
        arch_label = await page.inner_text("#anat-arch")
        type_label = await page.inner_text("#anat-type")
        surf_label = await page.inner_text("#anat-surf")
        print(f"[TEST 2] Tooth #16 Badge: Arch='{arch_label}', Type='{type_label}', Surface='{surf_label}'")
        assert "Maxillary Right" in arch_label or "Q1" in arch_label, "Arch label incorrect for #16!"
        assert "1st Molar" in type_label, "Type label incorrect for #16!"
        assert "occlusal_table" in surf_label, "Surface label incorrect for #16!"

        # Clear canvas points first
        await page.click("button:has-text('Clear')")
        await page.wait_for_timeout(300)

        # Scroll original photo into view
        img_locator = page.locator("#orig-img")
        await img_locator.scroll_into_view_if_needed()
        orig_img_box = await img_locator.bounding_box()
        print(f"[TEST 2] Original image bounding box: {orig_img_box}")

        # dental_1138 crack runs from approx (0.42, 0.47) to (0.56, 0.52)
        # Click 5 realistic crack points
        points_to_click = [
            (0.42, 0.47),
            (0.45, 0.48),
            (0.48, 0.49),
            (0.52, 0.50),
            (0.56, 0.52)
        ]
        for rel_x, rel_y in points_to_click:
            px = rel_x * orig_img_box["width"]
            py = rel_y * orig_img_box["height"]
            await img_locator.click(position={"x": px, "y": py})
            await page.wait_for_timeout(150)

        pts_count_text = await page.inner_text("#canvas-pts-count")
        print(f"[TEST 2] Traced points status: {pts_count_text}")
        assert "5" in pts_count_text, "Failed to register 5 traced points!"

        trace_snap_path = os.path.join(SCREENSHOTS_DIR, "snapshot_fix2_anatomical_badge_and_trace.png")
        await page.screenshot(path=trace_snap_path)
        print(f"[TEST 2] Captured trace & anatomical badge snapshot: {trace_snap_path}")

        # Click Render Manual 3D Carve
        print("[TEST 2] Clicking 'Render Manual 3D Carve' with 5 traced points...")
        await page.click("#manual-carve-btn")

        # Verify confirmation modal did NOT open
        await page.wait_for_timeout(500)
        modal_hidden_trace = await page.locator("#preset-confirm-modal").evaluate("el => el.classList.contains('hidden')")
        print(f"[TEST 2] Is modal hidden when traced points exist? {modal_hidden_trace} (Should be True)")
        assert modal_hidden_trace, "Modal should NOT appear when real photo points are traced!"

        # Wait for 3D render to complete
        await page.wait_for_selector("#carve-render-container:not(.hidden)", timeout=20000)
        await page.wait_for_timeout(1500)

        # Verify derivation banner and metrics reflect Photo-Derived
        banner_text_trace = await page.inner_text("#carve-derivation-banner")
        source_text_trace = await page.inner_text("#metric-source")
        target_surf_text = await page.inner_text("#metric-surface")
        verts_text = await page.inner_text("#metric-verts")

        print(f"[TEST 2] Trace Banner text: {banner_text_trace}")
        print(f"[TEST 2] Trace Metric source: {source_text_trace}")
        print(f"[TEST 2] Target surface: {target_surf_text}")
        print(f"[TEST 2] Influenced vertices: {verts_text}")

        assert "Photo-Derived" in banner_text_trace, "Banner did not indicate Photo-Derived!"
        assert "PHOTO-DERIVED" in banner_text_trace, "Banner badge missing PHOTO-DERIVED!"
        assert "Photo (5 pts)" in source_text_trace, "Metric source tile missing Photo (5 pts)!"
        assert "occlusal" in target_surf_text, "Target surface does not contain occlusal on molar #16!"

        carve_snap_path = os.path.join(SCREENSHOTS_DIR, "snapshot_fix2_photo_derived_render_tooth16.png")
        await page.screenshot(path=carve_snap_path)
        print(f"[TEST 2] Captured photo-derived 3D render snapshot: {carve_snap_path}")

        await browser.close()
        print("\n[SUCCESS] All verification tests passed flawlessly!")

if __name__ == "__main__":
    asyncio.run(run_verification())
