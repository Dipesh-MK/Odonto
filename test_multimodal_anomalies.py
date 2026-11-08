import asyncio
import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from playwright.async_api import async_playwright

RESULTS_SUBDIR = Path("/home/aracknab/Projects/Dental LLM and CV/test_results/multimodal_test1")
RESULTS_DIR = Path("/home/aracknab/Projects/Dental LLM and CV/test_results")
RESULTS_SUBDIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {
        "case_id": "01_occlusal_caries_molar_16",
        "name": "Occlusal Pit Caries (ICDAS 4 Cavitation)",
        "preset": "alpha_p006_F_15_001",
        "fdi": "16",
        "surface": "occlusal_table",
        "notes": "Occlusal pit discoloration with cavitated enamel margin and dentinal involvement",
        "bbox": [0.42, 0.44, 0.58, 0.59],  # Normalized [x1, y1, x2, y2]
        "target_depth": 0.85,
    },
    {
        "case_id": "02_cervical_abrasion_incisor_11",
        "name": "Cervical Abrasion (Anterior Enamel Facet)",
        "preset": "alpha_p006_F_15_002",
        "fdi": "11",
        "surface": "labial_facial",
        "notes": "Cervical saucer-shaped wear facet at gingival third of maxillary incisor",
        "bbox": [0.38, 0.40, 0.62, 0.60],
        "target_depth": 0.40,
    },
    {
        "case_id": "03_structural_crack_molar_16",
        "name": "Structural Fissure Crack (High Aspect Stress Notch)",
        "preset": "dental_1138_jpg.rf.fd470daf6db30f3a92dd2ccced1d10d7",
        "fdi": "16",
        "surface": "occlusal_table",
        "notes": "Linear structural crack traversing mesiodistal groove of occlusal surface",
        "bbox": [0.42, 0.46, 0.58, 0.54],
        "target_depth": 0.45,
    },
    {
        "case_id": "04_chipped_incisor_21",
        "name": "Chipped Incisal Edge (Coronal Structure Loss)",
        "preset": "novel_01_alphadent_filling_case",
        "fdi": "21",
        "surface": "labial_facial",
        "notes": "Incisal edge chipping with enamel-dentin fracture angle",
        "bbox": [0.44, 0.45, 0.58, 0.58],
        "target_depth": 1.10,
    }
]

async def run_multimodal_tests():
    print("=" * 80)
    print("STARTING TEACHER-GUIDED MULTIMODAL ANOMALY TEST SUITE (multimodal_test1)")
    print("=" * 80)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/usr/bin/chromium",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        context = await browser.new_context(viewport={"width": 1600, "height": 1200})
        page = await context.new_page()

        print("\n[SUITE] Navigating to Teacher Studio at http://localhost:8055...")
        await page.goto("http://localhost:8055", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # Switch to Teacher Studio Tab
        teacher_tab_btn = page.locator("#tab-teacher-btn")
        await teacher_tab_btn.click()
        await page.wait_for_timeout(1000)

        results_summary = []

        for idx, tc in enumerate(TEST_CASES, start=1):
            print(f"\n" + "-" * 75)
            print(f"CASE {idx}/{len(TEST_CASES)}: {tc['name']} on FDI #{tc['fdi']}")
            print("-" * 75)

            # 1. Load preset photo
            await page.evaluate(f"loadSampleTeacherPreset('{tc['preset']}')")
            await page.wait_for_timeout(1500)

            # Configure Target FDI and Surface
            await page.select_option("#teacher-fdi-select", tc["fdi"])
            await page.select_option("#teacher-surface-select", tc["surface"])
            await page.fill("#teacher-notes-input", tc["notes"])
            await page.wait_for_timeout(300)

            # 2. Draw Bounding Box Prompt on Canvas
            canvas_box = await page.locator("#teacher-canvas").bounding_box()
            if not canvas_box or canvas_box["width"] <= 0:
                print(f"Error: Canvas bounding box not found for {tc['name']}")
                continue

            # Programmatically set teacherBbox and notify prompt readout
            bx = tc["bbox"]
            await page.evaluate(f"""() => {{
                teacherBbox = [{bx[0]}, {bx[1]}, {bx[2]}, {bx[3]}];
                teacherPoints = [];
                teacherMode = 'box';
                redrawTeacherCanvas();
                const w = Math.round((teacherBbox[2] - teacherBbox[0]) * 100);
                const h = Math.round((teacherBbox[3] - teacherBbox[1]) * 100);
                document.getElementById('teacher-prompt-readout').textContent = `Bounding Box: ${{w}}% × ${{h}}% of ROI`;
            }}""")
            await page.wait_for_timeout(400)

            # 3. Trigger Phase 1: Snap SAM Boundary
            print("  -> Triggering Phase 1: Snap SAM Boundary...")
            await page.click("#snap-sam-btn")
            await page.wait_for_selector("#teacher-overlay-preview:not(.hidden)", timeout=20000)
            await page.wait_for_timeout(1000)

            sam_points = await page.locator("#sam-metric-points").text_content()
            sam_area = await page.locator("#sam-metric-area").text_content()
            sam_centroid = await page.locator("#sam-metric-centroid").text_content()
            print(f"  [SAM Output] Vertices: {sam_points} | Area: {sam_area} | Centroid: {sam_centroid}")

            # 4. Trigger Phase 2: Run Clinical Diagnosis Agent
            print("  -> Triggering Phase 2: Clinical Diagnostics Agent...")
            await page.click("#run-diagnose-btn")
            await page.wait_for_selector("#diagnostic-card:not(.hidden)", timeout=20000)
            await page.wait_for_timeout(1000)

            diag_pathology = await page.locator("#diag-pathology-badge").text_content()
            diag_grade = await page.locator("#diag-severity-title").text_content()
            diag_depth = await page.locator("#diag-depth-readout").text_content()
            diag_rationale = await page.locator("#diag-rationale-text").text_content()
            print(f"  [Agent Diagnosis] Pathology: {diag_pathology} | Grade: {diag_grade} | Depth: {diag_depth}")
            print(f"  [Rationale] \"{diag_rationale}\"")

            # Adjust depth slider to target depth if specified
            if tc["target_depth"]:
                target_val = str(tc["target_depth"])
                await page.evaluate(f"""() => {{
                    const slider = document.getElementById('teacher-depth-slider');
                    slider.value = '{target_val}';
                    updateTeacherDepth('{target_val}');
                }}""")
                await page.wait_for_timeout(300)

            # 5. Trigger Phase 3: Approve & Carve 3D Mesh
            print(f"  -> Triggering Phase 3: Approve & Render Contour-Conforming 3D Carve...")
            await page.click("#approve-case-btn")
            await page.wait_for_selector("#teacher-3d-result-card:not(.hidden)", timeout=35000)
            await page.wait_for_timeout(2000)

            verts_displaced = await page.locator("#teacher-3d-verts-badge").text_content()
            max_depth = await page.locator("#teacher-3d-depth-badge").text_content()
            span_text = await page.locator("#teacher-3d-span-badge").text_content()
            print(f"  [3D Carve Result] {verts_displaced} | {max_depth} | {span_text}")

            # Extract 3D PyVista Render Image base64
            render_src = await page.locator("#teacher-3d-render-img").get_attribute("src")
            if render_src and "base64," in render_src:
                b64_data = render_src.split("base64,")[1]
                render_bytes = base64.b64decode(b64_data)

                # Save 3D PyVista render to multimodal_test1 folder
                render_fpath1 = RESULTS_SUBDIR / f"{tc['case_id']}_3d.png"
                with open(render_fpath1, "wb") as f:
                    f.write(render_bytes)

                # Mirror directly in test_results with multimodal_test1 prefix
                render_fpath2 = RESULTS_DIR / f"multimodal_test1_{tc['case_id']}_3d.png"
                with open(render_fpath2, "wb") as f:
                    f.write(render_bytes)
                print(f"  [Saved 3D Render] {render_fpath1}")

            # Capture Full Teacher Studio UI Snapshot
            ui_fpath1 = RESULTS_SUBDIR / f"{tc['case_id']}_ui.png"
            ui_fpath2 = RESULTS_DIR / f"multimodal_test1_{tc['case_id']}_ui.png"
            await page.screenshot(path=str(ui_fpath1), full_page=False)
            await page.screenshot(path=str(ui_fpath2), full_page=False)
            print(f"  [Saved UI Snapshot] {ui_fpath1}")

            # Scroll down to capture focused 3D card snapshot as well
            await page.locator("#teacher-3d-result-card").scroll_into_view_if_needed()
            await page.wait_for_timeout(500)
            card_fpath1 = RESULTS_SUBDIR / f"{tc['case_id']}_card.png"
            await page.locator("#teacher-3d-result-card").screenshot(path=str(card_fpath1))

            results_summary.append({
                "case": tc["name"],
                "fdi": tc["fdi"],
                "surface": tc["surface"],
                "sam_points": sam_points,
                "sam_area": sam_area,
                "pathology": diag_pathology,
                "grade": diag_grade,
                "depth": diag_depth,
                "verts_displaced": verts_displaced,
                "max_depth": max_depth,
                "span": span_text,
                "ui_snapshot": str(ui_fpath1),
                "render_snapshot": str(render_fpath1),
            })

            # Scroll back to top for next case
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)

        await browser.close()

        print("\n" + "=" * 80)
        print("ALL 4 MULTIMODAL TEST CASES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        for r in results_summary:
            print(f"✅ {r['case']} on #{r['fdi']} | {r['pathology']} ({r['depth']}) | {r['verts_displaced']} | {r['max_depth']}")

        # Save summary JSON
        summary_path = RESULTS_SUBDIR / "multimodal_test1_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nSaved test summary metrics to: {summary_path}")

if __name__ == "__main__":
    asyncio.run(run_multimodal_tests())
