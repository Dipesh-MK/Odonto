import asyncio
import os
import json
import base64
from pathlib import Path
from playwright.async_api import async_playwright

RESULTS_DIR = Path("/home/aracknab/Projects/Dental LLM and CV/test_results/side_by_side_comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {
        "case_id": "case1_occlusal_caries_molar_16",
        "name": "Occlusal Pit Caries (ICDAS 4 Cavitation)",
        "preset": "alpha_p006_F_15_001",
        "fdi": "16",
        "surface": "occlusal_table",
        "notes": "Occlusal pit discoloration with cavitated enamel margin and dentinal involvement",
        "bbox": [0.42, 0.44, 0.58, 0.59],
        "target_depth": 0.75,
    },
    {
        "case_id": "case2_cervical_abrasion_incisor_11",
        "name": "Cervical Abrasion (Anterior Enamel Facet)",
        "preset": "alpha_p006_F_15_002",
        "fdi": "11",
        "surface": "labial_facial",
        "notes": "Cervical saucer-shaped wear facet at gingival third of maxillary central incisor",
        "bbox": [0.38, 0.40, 0.62, 0.60],
        "target_depth": 0.40,
    },
    {
        "case_id": "case3_structural_crack_molar_16",
        "name": "Structural Fissure Crack (High Aspect Stress Notch)",
        "preset": "dental_1138_jpg.rf.fd470daf6db30f3a92dd2ccced1d10d7",
        "fdi": "16",
        "surface": "occlusal_table",
        "notes": "Linear structural crack traversing mesiodistal groove of occlusal surface",
        "bbox": [0.42, 0.46, 0.58, 0.54],
        "target_depth": 0.45,
    },
    {
        "case_id": "case4_chipped_incisor_21",
        "name": "Chipped Incisal Edge (Coronal Structure Loss)",
        "preset": "novel_01_alphadent_filling_case",
        "fdi": "21",
        "surface": "labial_facial",
        "notes": "Incisal edge chipping with enamel-dentin fracture angle",
        "bbox": [0.44, 0.45, 0.58, 0.58],
        "target_depth": 1.10,
    }
]

async def run_side_by_side_suite():
    print("=" * 85)
    print("TESTING METHOD A (PARAMETRIC) vs METHOD B (DEPTH ANYTHING V2) ACROSS MULTIPLE ANOMALIES")
    print("=" * 85)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path="/usr/bin/chromium",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        context = await browser.new_context(viewport={"width": 1600, "height": 1300})
        page = await context.new_page()

        print("\n[STEP 1] Navigating to Teacher Studio at http://localhost:8055...")
        await page.goto("http://localhost:8055", wait_until="networkidle")
        await page.wait_for_timeout(2000)

        # Switch to Teacher Studio tab
        await page.click("#tab-teacher-btn")
        await page.wait_for_timeout(1000)

        all_summaries = []

        for idx, tc in enumerate(TEST_CASES, start=1):
            print("\n" + "=" * 80)
            print(f"CASE {idx}/{len(TEST_CASES)}: {tc['name']} on Target Tooth FDI #{tc['fdi']} ({tc['surface']})")
            print("=" * 80)

            # Step A: Load Preset Photo
            print(f"  [1] Loading Preset: {tc['preset']}...")
            await page.evaluate(f"loadSampleTeacherPreset('{tc['preset']}')")
            await page.wait_for_timeout(1500)

            # Set Target FDI, Surface, and Notes
            await page.select_option("#teacher-fdi-select", tc["fdi"])
            await page.select_option("#teacher-surface-select", tc["surface"])
            await page.fill("#teacher-notes-input", tc["notes"])
            await page.wait_for_timeout(300)

            # Step B: Set Prompt Bounding Box on Canvas
            bx = tc["bbox"]
            print(f"  [2] Setting Prompt Bounding Box: {bx}...")
            await page.evaluate(f"""() => {{
                teacherBbox = [{bx[0]}, {bx[1]}, {bx[2]}, {bx[3]}];
                teacherPoints = [];
                teacherMode = 'box';
                redrawTeacherCanvas();
                const w = Math.round((teacherBbox[2] - teacherBbox[0]) * 100);
                const h = Math.round((teacherBbox[3] - teacherBbox[1]) * 100);
                document.getElementById('teacher-prompt-readout').textContent = `Bounding Box: ${{w}}% × ${{h}}% of ROI`;
            }}""")
            await page.wait_for_timeout(500)

            # Step C: Snap SAM Boundary + Extract Depth Anything V2 Heightmap
            print("  [3] Snapping SAM Boundary & Extracting Depth Anything V2 Heightmap...")
            await page.click("#snap-sam-btn")
            await page.wait_for_selector("#teacher-overlay-preview:not(.hidden)", timeout=35000)
            await page.wait_for_selector("#teacher-heatmap-preview:not(.hidden)", timeout=35000)
            await page.wait_for_timeout(1500)

            sam_pts = await page.locator("#sam-metric-points").text_content()
            sam_area = await page.locator("#sam-metric-area").text_content()
            depth_samples = await page.locator("#depth-metric-samples").text_content()
            print(f"      SAM Boundary: {sam_pts} | Enclosed Area: {sam_area}")
            print(f"      Depth Anything V2 Detrending Baseline Samples: {depth_samples}")

            # Step D: Run Clinical Diagnostics & Transfer Agent
            print("  [4] Running Clinical Diagnosis & Target Recommendation Agent...")
            await page.click("#run-diagnose-btn")
            await page.wait_for_selector("#diagnostic-card:not(.hidden)", timeout=25000)
            await page.wait_for_selector("#agent-recommendation-box:not(.hidden)", timeout=25000)
            await page.wait_for_timeout(1500)

            diag_pathology = await page.locator("#diag-pathology-badge").text_content()
            diag_grade = await page.locator("#diag-severity-title").text_content()
            agent_rec_fdi = await page.locator("#agent-rec-fdi-badge").text_content()
            agent_rationale = await page.locator("#agent-rec-rationale").text_content()
            print(f"      Pathology: {diag_pathology} ({diag_grade})")
            print(f"      Agent Recommended Tooth: {agent_rec_fdi} | Rationale: {agent_rationale}")

            # Optionally adjust depth slider
            if tc.get("target_depth"):
                val_str = str(tc["target_depth"])
                await page.evaluate(f"""() => {{
                    const slider = document.getElementById('teacher-depth-slider');
                    slider.value = '{val_str}';
                    updateTeacherDepth('{val_str}');
                }}""")
                await page.wait_for_timeout(300)

            # Step E: Capture Teacher Studio UI (SAM + Depth Heatmap + Agent Card)
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(400)
            studio_snap = RESULTS_DIR / f"{tc['case_id']}_studio_ui.png"
            await page.screenshot(path=str(studio_snap), full_page=False)
            print(f"      Saved Studio UI Snapshot: {studio_snap.name}")

            # Step F: Select Side-by-Side Comparison Mode
            print("  [5] Activating '⚡ Side-by-Side' Comparison Mode...")
            await page.click("#btn-method-compare")
            await page.wait_for_timeout(400)

            # Step G: Execute Dual Carve
            print("  [6] Executing Dual 3D Carve (Method A vs Method B)...")
            await page.click("#approve-case-btn")
            await page.wait_for_selector("#side-by-side-comparison-view:not(.hidden)", timeout=45000)
            await page.wait_for_timeout(2500)

            verts_a = await page.locator("#compare-verts-a").text_content()
            depth_a = await page.locator("#compare-depth-a").text_content()
            verts_b = await page.locator("#compare-verts-b").text_content()
            depth_b = await page.locator("#compare-depth-b").text_content()

            print(f"      Method A (Parametric Mathematical): {verts_a} | Max Depth: {depth_a}")
            print(f"      Method B (Depth Anything V2):       {verts_b} | Max Depth: {depth_b}")

            # Step H: Scroll to Side-by-Side Result Card and capture screenshot
            await page.locator("#teacher-3d-result-card").scroll_into_view_if_needed()
            await page.wait_for_timeout(1000)

            sbs_card_snap = RESULTS_DIR / f"{tc['case_id']}_side_by_side_card.png"
            await page.locator("#teacher-3d-result-card").screenshot(path=str(sbs_card_snap))
            print(f"      Saved Side-by-Side Result Card: {sbs_card_snap.name}")

            # Extract individual renders
            render_a_path = RESULTS_DIR / f"{tc['case_id']}_method_a_render.png"
            src_a = await page.locator("#compare-render-a").get_attribute("src")
            if src_a and "base64," in src_a:
                with open(render_a_path, "wb") as f:
                    f.write(base64.b64decode(src_a.split("base64,")[1]))
                print(f"      Saved Method A 3D Render: {render_a_path.name}")

            render_b_path = RESULTS_DIR / f"{tc['case_id']}_method_b_render.png"
            src_b = await page.locator("#compare-render-b").get_attribute("src")
            if src_b and "base64," in src_b:
                with open(render_b_path, "wb") as f:
                    f.write(base64.b64decode(src_b.split("base64,")[1]))
                print(f"      Saved Method B 3D Render: {render_b_path.name}")

            # Record metrics
            all_summaries.append({
                "case_id": tc["case_id"],
                "anomaly_name": tc["name"],
                "target_fdi": tc["fdi"],
                "target_surface": tc["surface"],
                "notes": tc["notes"],
                "sam_points": sam_pts,
                "sam_area": sam_area,
                "depth_enamel_samples": depth_samples,
                "pathology": diag_pathology,
                "severity": diag_grade,
                "agent_recommended_fdi": agent_rec_fdi,
                "agent_rationale": agent_rationale,
                "method_a": {
                    "name": "Method A (Synthetic Parametric)",
                    "displaced_vertices": verts_a,
                    "max_depth": depth_a,
                    "render_file": str(render_a_path),
                },
                "method_b": {
                    "name": "Method B (Monocular Depth Relief)",
                    "displaced_vertices": verts_b,
                    "max_depth": depth_b,
                    "render_file": str(render_b_path),
                },
                "ui_snapshot": str(studio_snap),
                "side_by_side_card": str(sbs_card_snap)
            })

            # Scroll back to top for next case
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)

        await browser.close()

        # Save complete summary JSON
        summary_path = RESULTS_DIR / "side_by_side_all_anomalies_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)

        print("\n" + "=" * 85)
        print("ALL 4 ANOMALY CASES TESTED SIDE-BY-SIDE SUCCESSFULLY!")
        print("=" * 85)
        for s in all_summaries:
            print(f"✅ {s['anomaly_name']} on FDI #{s['target_fdi']}:")
            print(f"   - Method A: {s['method_a']['displaced_vertices']} | {s['method_a']['max_depth']}")
            print(f"   - Method B: {s['method_b']['displaced_vertices']} | {s['method_b']['max_depth']}")
        print(f"\nSaved metrics summary to: {summary_path}")

if __name__ == "__main__":
    asyncio.run(run_side_by_side_suite())
