import asyncio
import os
from playwright.async_api import async_playwright

RESULTS_DIR = "/home/aracknab/Projects/Dental LLM and CV/test_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

async def run_comprehensive_tests():
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/usr/bin/chromium",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        context = await browser.new_context(viewport={"width": 1440, "height": 1100})
        page = await context.new_page()

        print("[TEST SUITE] Navigating to http://localhost:8055...")
        await page.goto("http://localhost:8055", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        carve_btn = page.locator("#carve-btn")
        manual_carve_btn = page.locator("#manual-carve-btn")
        img_locator = page.locator("#orig-img")

        # =====================================================================
        # CASE 1: Autonomous Caries on Mandibular Molar FDI #36
        # =====================================================================
        print("\n--- CASE 1: Autonomous Caries on Molar #36 ---")
        # Uncheck abrasion so Caries is selected directly
        if await page.is_checked("#toggle-abrasion"):
            await page.uncheck("#toggle-abrasion")
            await page.wait_for_timeout(1000)

        await page.select_option("#preset-select", "alpha_p025_F_32_001")
        await page.wait_for_selector("#loading-spinner", state="hidden", timeout=20000)
        await page.wait_for_timeout(1500)

        await page.select_option("#fdi-select", "36")
        await page.wait_for_timeout(500)

        await carve_btn.scroll_into_view_if_needed()
        await carve_btn.click()
        await page.wait_for_selector("#carve-render-container:not(.hidden)", timeout=20000)
        await page.wait_for_timeout(1500)

        # Scroll to 3D render container for screenshot
        await page.locator("#carve-render-container").scroll_into_view_if_needed()
        c1_snap = os.path.join(RESULTS_DIR, "01_caries_molar_36_autonomous.png")
        await page.screenshot(path=c1_snap)
        print(f"[CASE 1] Saved: {c1_snap}")

        # =====================================================================
        # CASE 2: Autonomous Abrasion on Maxillary Incisor FDI #11
        # =====================================================================
        print("\n--- CASE 2: Autonomous Abrasion on Incisor #11 ---")
        if not await page.is_checked("#toggle-abrasion"):
            await page.check("#toggle-abrasion")
            await page.wait_for_timeout(1000)

        await page.select_option("#preset-select", "alpha_p006_F_15_002")
        await page.wait_for_selector("#loading-spinner", state="hidden", timeout=20000)
        await page.wait_for_timeout(1500)

        await page.select_option("#fdi-select", "11")
        await page.wait_for_timeout(500)

        await carve_btn.scroll_into_view_if_needed()
        await carve_btn.click()
        await page.wait_for_selector("#carve-render-container:not(.hidden)", timeout=20000)
        await page.wait_for_timeout(1500)

        await page.locator("#carve-render-container").scroll_into_view_if_needed()
        c2_snap = os.path.join(RESULTS_DIR, "02_abrasion_incisor_11_autonomous.png")
        await page.screenshot(path=c2_snap)
        print(f"[CASE 2] Saved: {c2_snap}")

        # =====================================================================
        # CASE 3: Whitelist Safety Block — Posterior Abrasion on Premolar #34
        # =====================================================================
        print("\n--- CASE 3: Whitelist Safety Block on Premolar #34 ---")
        await page.select_option("#fdi-select", "34")
        await page.wait_for_timeout(500)

        await carve_btn.scroll_into_view_if_needed()
        await carve_btn.click()
        await page.wait_for_selector("#carve-block-alert:not(.hidden)", timeout=10000)
        await page.wait_for_timeout(1000)

        await page.locator("#carve-block-alert").scroll_into_view_if_needed()
        c3_snap = os.path.join(RESULTS_DIR, "03_whitelist_block_posterior_abrasion_premolar_34.png")
        await page.screenshot(path=c3_snap)
        print(f"[CASE 3] Saved: {c3_snap}")

        # =====================================================================
        # CASE 4: Crack Autonomous Block / 2D-Only Warning on FDI #16
        # =====================================================================
        print("\n--- CASE 4: Crack Autonomous 2D-Only Warning on FDI #16 ---")
        await page.select_option("#preset-select", "dental_1138_jpg.rf.fd470daf6db30f3a92dd2ccced1d10d7")
        await page.wait_for_selector("#loading-spinner", state="hidden", timeout=20000)
        await page.wait_for_timeout(1500)

        await page.select_option("#fdi-select", "16")
        await page.wait_for_timeout(500)

        await carve_btn.scroll_into_view_if_needed()
        await carve_btn.click()
        await page.wait_for_selector("#carve-crack-notice:not(.hidden)", timeout=10000)
        await page.wait_for_timeout(1000)

        await page.locator("#carve-crack-notice").scroll_into_view_if_needed()
        c4_snap = os.path.join(RESULTS_DIR, "04_crack_autonomous_2d_only_blocked.png")
        await page.screenshot(path=c4_snap)
        print(f"[CASE 4] Saved: {c4_snap}")

        # =====================================================================
        # CASE 5: Manual Photo-Derived Crack Carve on Maxillary Molar FDI #16
        # =====================================================================
        print("\n--- CASE 5: Manual Photo-Derived Crack on Molar #16 ---")
        await page.select_option("#manual-profile-select", "v_groove_path")
        await page.click("button:has-text('Clear')")
        await page.wait_for_timeout(300)

        await img_locator.scroll_into_view_if_needed()
        img_box = await img_locator.bounding_box()

        # Click 5 crack points on molar occlusal crack
        pts_molar = [(0.42, 0.47), (0.45, 0.48), (0.48, 0.49), (0.52, 0.50), (0.56, 0.52)]
        for rx, ry in pts_molar:
            await img_locator.click(position={"x": rx * img_box["width"], "y": ry * img_box["height"]})
            await page.wait_for_timeout(120)

        await manual_carve_btn.scroll_into_view_if_needed()
        await manual_carve_btn.click()

        await page.wait_for_selector("#carve-render-container:not(.hidden)", timeout=20000)
        await page.wait_for_timeout(1500)

        await page.locator("#carve-render-container").scroll_into_view_if_needed()
        c5_snap = os.path.join(RESULTS_DIR, "05_photo_derived_crack_maxillary_molar_16.png")
        await page.screenshot(path=c5_snap)
        print(f"[CASE 5] Saved: {c5_snap}")

        # =====================================================================
        # CASE 6: Manual Photo-Derived Crack on Anterior Central Incisor FDI #21
        # =====================================================================
        print("\n--- CASE 6: Manual Photo-Derived Crack on Incisor #21 ---")
        await page.select_option("#preset-select", "dental_1282_jpg.rf.e0b8c616bbb750f4e4d63d90475ad90d")
        await page.wait_for_selector("#loading-spinner", state="hidden", timeout=20000)
        await page.wait_for_timeout(1500)

        # Select Upper Left Central Incisor #21
        await page.select_option("#fdi-select", "21")
        await page.wait_for_timeout(500)

        await page.select_option("#manual-profile-select", "v_groove_path")
        await page.click("button:has-text('Clear')")
        await page.wait_for_timeout(300)

        await img_locator.scroll_into_view_if_needed()
        img_box2 = await img_locator.bounding_box()

        # Trace vertical hairline fracture along incisal to cervical path
        pts_incisor = [(0.50, 0.35), (0.51, 0.43), (0.50, 0.52), (0.49, 0.60)]
        for rx, ry in pts_incisor:
            await img_locator.click(position={"x": rx * img_box2["width"], "y": ry * img_box2["height"]})
            await page.wait_for_timeout(120)

        await manual_carve_btn.scroll_into_view_if_needed()
        await manual_carve_btn.click()

        await page.wait_for_selector("#carve-render-container:not(.hidden)", timeout=20000)
        await page.wait_for_timeout(1500)

        await page.locator("#carve-render-container").scroll_into_view_if_needed()
        c6_snap = os.path.join(RESULTS_DIR, "06_photo_derived_crack_incisor_21.png")
        await page.screenshot(path=c6_snap)
        print(f"[CASE 6] Saved: {c6_snap}")

        # =====================================================================
        # CASE 7: Manual Chipped Cusp on Mandibular Premolar FDI #34
        # =====================================================================
        print("\n--- CASE 7: Manual Chipped Cusp on Premolar #34 ---")
        await page.select_option("#fdi-select", "34")
        await page.select_option("#manual-profile-select", "face_removal_recap")
        await page.select_option("#chip-cusp-select", "buccal")
        await page.fill("#chip-height-input", "1.6")
        await page.fill("#chip-angle-input", "40.0")

        await manual_carve_btn.scroll_into_view_if_needed()
        await manual_carve_btn.click()

        await page.wait_for_selector("#carve-render-container:not(.hidden)", timeout=20000)
        await page.wait_for_timeout(1500)

        await page.locator("#carve-render-container").scroll_into_view_if_needed()
        c7_snap = os.path.join(RESULTS_DIR, "07_manual_chipped_buccal_cusp_premolar_34.png")
        await page.screenshot(path=c7_snap)
        print(f"[CASE 7] Saved: {c7_snap}")

        # =====================================================================
        # CASE 8: Manual Chipped Cusp on Mandibular Molar FDI #36
        # =====================================================================
        print("\n--- CASE 8: Manual Chipped Cusp on Molar #36 ---")
        await page.select_option("#fdi-select", "36")
        await page.select_option("#manual-profile-select", "face_removal_recap")
        await page.select_option("#chip-cusp-select", "mesiobuccal")
        await page.fill("#chip-height-input", "1.8")
        await page.fill("#chip-angle-input", "35.0")

        await manual_carve_btn.scroll_into_view_if_needed()
        await manual_carve_btn.click()

        await page.wait_for_selector("#carve-render-container:not(.hidden)", timeout=20000)
        await page.wait_for_timeout(1500)

        await page.locator("#carve-render-container").scroll_into_view_if_needed()
        c8_snap = os.path.join(RESULTS_DIR, "08_manual_chipped_mesiobuccal_cusp_molar_36.png")
        await page.screenshot(path=c8_snap)
        print(f"[CASE 8] Saved: {c8_snap}")

        # =====================================================================
        # CASE 9: Batch Grid Testing
        # =====================================================================
        print("\n--- CASE 9: Batch Grid Testing ---")
        batch_tab_btn = page.locator("#tab-batch-btn")
        await batch_tab_btn.scroll_into_view_if_needed()
        await batch_tab_btn.click()
        await page.wait_for_timeout(500)

        run_batch_btn = page.locator("button:has-text('Run 8-Sample Benchmark Grid')")
        await run_batch_btn.click()

        # Wait for all cards to be populated
        await page.wait_for_selector("#batch-grid .bg-slate-900", timeout=30000)
        await page.wait_for_timeout(2000)

        c9_snap = os.path.join(RESULTS_DIR, "09_batch_grid_8_samples.png")
        await page.screenshot(path=c9_snap)
        print(f"[CASE 9] Saved: {c9_snap}")

        await browser.close()
        print("\n[ALL 9 TEST CASES COMPLETED SUCCESSFULLY!]")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_tests())
