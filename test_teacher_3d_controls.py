import sys
import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: console_errors.append(str(err)))

        print("Navigating to http://127.0.0.1:5000/ ...")
        await page.goto("http://127.0.0.1:5000/", wait_until="networkidle")

        # 1. Login as teacher
        print("Logging in as teacher@dental.edu...")
        await page.fill("#auth-email", "teacher@dental.edu")
        await page.fill("#auth-password", "teacher123")
        await page.click("#auth-submit-btn")
        await page.wait_for_timeout(1000)

        role = await page.evaluate("CURRENT_USER?.role")
        print("Current user role:", role)
        assert role == "teacher", f"Expected teacher, got {role}"

        # 2. Navigate to SAM 2D->3D Studio (sam2-pipeline)
        print("Switching to sam2-pipeline tab...")
        await page.evaluate("switchTab('sam2-pipeline')")
        await page.wait_for_timeout(600)

        # 3. Load clean tooth preview
        print("Loading clean tooth preview...")
        await page.evaluate("loadCleanToothPreview()")
        await page.wait_for_timeout(2000)

        hud_text = await page.inner_text("#pipe-3d-hud")
        print("Pipe HUD text:", hud_text)
        assert "Clean Tooth: FDI #" in hud_text, f"Unexpected hud: {hud_text}"

        mesh_exists = await page.evaluate("Boolean(pipeThree.mesh)")
        print("pipeThree mesh loaded:", mesh_exists)
        assert mesh_exists, "pipeThree.mesh is null!"

        # 4. Measure camera position before drag
        initial_cam = await page.evaluate("""() => {
            return {
                pos: pipeThree.camera.position,
                target: pipeThree.controls.target,
                azimuth: pipeThree.controls.getAzimuthalAngle()
            };
        }""")
        print("Initial pipe camera:", initial_cam)

        # 5. Scroll pipe canvas into view and drag from center towards right (+150px)
        pipe_canvas = page.locator("#pipe-three-container canvas")
        await pipe_canvas.scroll_into_view_if_needed()
        canvas_box = await pipe_canvas.bounding_box()
        print("Canvas box:", canvas_box)
        cx = canvas_box["x"] + canvas_box["width"] / 2
        cy = canvas_box["y"] + canvas_box["height"] / 2

        print(f"Dragging on pipe canvas from ({cx}, {cy}) to ({cx + 150}, {cy})...")
        await page.mouse.move(cx, cy)
        await page.mouse.down()
        for step in range(1, 15):
            await page.mouse.move(cx + (150 * step / 14), cy)
            await page.wait_for_timeout(20)
        await page.mouse.up()
        await page.wait_for_timeout(500)

        dragged_cam = await page.evaluate("""() => {
            return {
                pos: pipeThree.camera.position,
                target: pipeThree.controls.target,
                azimuth: pipeThree.controls.getAzimuthalAngle()
            };
        }""")
        print("Post-drag pipe camera:", dragged_cam)
        az_diff = dragged_cam["azimuth"] - initial_cam["azimuth"]
        print(f"Azimuth change: {az_diff:.4f} rad")
        assert abs(az_diff) > 0.1, f"Azimuth should have changed noticeably! az_diff={az_diff}"

        # 6. Test perspective view buttons: Occlusal, Labial, Reset
        print("Testing Occlusal view button...")
        await page.evaluate("setPipeline3DView('top')")
        await page.wait_for_timeout(300)
        top_cam = await page.evaluate("pipeThree.camera.position")
        print("Top view position:", top_cam)

        print("Testing Labial view button...")
        await page.evaluate("setPipeline3DView('front')")
        await page.wait_for_timeout(300)
        front_cam = await page.evaluate("pipeThree.camera.position")
        print("Front view position:", front_cam)

        print("Testing Reset view button...")
        await page.evaluate("setPipeline3DView('reset')")
        await page.wait_for_timeout(300)

        # Screenshot pipe preview
        await pipe_canvas.screenshot(path="teacher_pipe_preview_verified.png")
        print("Saved teacher_pipe_preview_verified.png")

        # 7. Test 3D Viewer & Sim (simThree)
        print("Switching to case-library and selecting a case for 3D Viewer & Sim...")
        await page.evaluate("switchTab('case-library')")
        await page.wait_for_timeout(1000)

        # Select first case
        first_case_uid = await page.evaluate("ROOM_CASES[0]?.case_uid")
        print("First case UID:", first_case_uid)
        if first_case_uid:
            await page.evaluate(f"selectActiveCase('{first_case_uid}')")
            await page.wait_for_timeout(1000)

            print("Switching to 3d-viewer-and-sim tab...")
            await page.evaluate("switchTab('3d-viewer-and-sim')")
            await page.wait_for_timeout(1000)

            sim_mesh_exists = await page.evaluate("Boolean(simThree.mesh)")
            print("simThree mesh loaded:", sim_mesh_exists)
            assert sim_mesh_exists, "simThree.mesh should be loaded!"

            sim_initial = await page.evaluate("""() => {
                return {
                    pos: simThree.camera.position,
                    target: simThree.controls.target,
                    azimuth: simThree.controls.getAzimuthalAngle()
                };
            }""")
            print("Initial sim camera:", sim_initial)

            sim_canvas = page.locator("#sim-threejs-container canvas")
            await sim_canvas.scroll_into_view_if_needed()
            sim_box = await sim_canvas.bounding_box()
            scx = sim_box["x"] + sim_box["width"] / 2
            scy = sim_box["y"] + sim_box["height"] / 2

            print(f"Dragging on sim canvas from ({scx}, {scy}) to ({scx + 150}, {scy})...")
            await page.mouse.move(scx, scy)
            await page.mouse.down()
            for step in range(1, 15):
                await page.mouse.move(scx + (150 * step / 14), scy)
                await page.wait_for_timeout(20)
            await page.mouse.up()
            await page.wait_for_timeout(500)

            sim_dragged = await page.evaluate("""() => {
                return {
                    pos: simThree.camera.position,
                    target: simThree.controls.target,
                    azimuth: simThree.controls.getAzimuthalAngle()
                };
            }""")
            print("Post-drag sim camera:", sim_dragged)
            sim_diff = sim_dragged["azimuth"] - sim_initial["azimuth"]
            print(f"Sim Azimuth change: {sim_diff:.4f} rad")
            assert abs(sim_diff) > 0.1, f"Sim Azimuth should have changed noticeably! sim_diff={sim_diff}"

            # Test sim perspective view buttons
            print("Testing Reset Sim camera...")
            await page.evaluate("resetSimCamera()")
            await page.wait_for_timeout(300)
            print("Testing Top Sim camera...")
            await page.evaluate("setSimCameraTop()")
            await page.wait_for_timeout(300)
            print("Resetting Sim camera...")
            await page.evaluate("resetSimCamera()")
            await page.wait_for_timeout(300)

            await sim_canvas.screenshot(path="teacher_sim_preview_verified.png")
            print("Saved teacher_sim_preview_verified.png")

        print("Console errors:", [e for e in console_errors if "favicon" not in e])
        assert len([e for e in console_errors if "favicon" not in e]) == 0, f"Unexpected errors: {console_errors}"
        print("ALL TEACHER 3D VIEWPORT CONTROLS VERIFIED SUCCESSFULLY!")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
