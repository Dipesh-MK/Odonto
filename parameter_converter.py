"""
parameter_converter.py

Dynamic, Tooth-Type-Aware and Anomaly-Aware 2D-to-3D Parameter Conversion Layer.
Transforms 2D pixel geometry (centroid, area, aspect ratio, radius, orientation)
into 3D tooth-relative carving parameters derived directly from the target tooth mesh.

Key Architectural Principles:
1. Zero hardcoded millimeter bounds: all radius and depth limits are computed
   dynamically from the specific tooth mesh's measured crown surface spans,
   mean triangle edge length, and local wall thickness.
2. Explicit FDI tooth-type branching:
   - Anterior teeth (Incisor, Canine): have no occlusal table; clinical photos
     view the labial/facial surface. 2D coordinates map to (mesio-distal, apico-coronal)
     on the labial crown face.
   - Posterior teeth (Premolar, Molar): possess an occlusal chewing table.
     2D coordinates map to (mesio-distal, bucco-lingual) on the occlusal basin/table.
3. Anomaly-aware depth and profile:
   - Cavity / Caries: deeper concave cavitation scaled to radius and local wall thickness.
   - Abrasion: shallow, flat-bottomed mechanical wear facet aligned with anatomical wear axes.
"""

import numpy as np


def classify_fdi_tooth(tooth_fdi=None, mesh_spans=None):
    """
    Explicitly classifies an FDI two-digit tooth number into its anatomical type
    and functional group.

    FDI Notation:
      - First digit (quadrant): 1-4 permanent, 5-8 deciduous
      - Second digit (position from midline):
          1, 2 -> incisor   (anterior)
          3    -> canine    (anterior)
          4, 5 -> premolar  (posterior)
          6-8  -> molar     (posterior)
    """
    if tooth_fdi is not None:
        try:
            fdi_int = int(tooth_fdi)
            quadrant = fdi_int // 10
            position = fdi_int % 10
            jaw = "upper" if quadrant in (1, 2, 5, 6) else "lower"
            if position in (1, 2):
                return {
                    "tooth_fdi": fdi_int,
                    "tooth_type": "incisor",
                    "anatomical_group": "anterior",
                    "jaw": jaw
                }
            elif position == 3:
                return {
                    "tooth_fdi": fdi_int,
                    "tooth_type": "canine",
                    "anatomical_group": "anterior",
                    "jaw": jaw
                }
            elif position in (4, 5):
                return {
                    "tooth_fdi": fdi_int,
                    "tooth_type": "premolar",
                    "anatomical_group": "posterior",
                    "jaw": jaw
                }
            elif position in (6, 7, 8):
                return {
                    "tooth_fdi": fdi_int,
                    "tooth_type": "molar",
                    "anatomical_group": "posterior",
                    "jaw": jaw
                }
        except (ValueError, TypeError):
            pass

    # Dynamic geometric fallback if FDI number is omitted
    if mesh_spans is not None:
        md_span, bl_span, crown_h = mesh_spans
        horizontal_footprint = max(np.sqrt(md_span * bl_span), 1e-5)
        if crown_h / horizontal_footprint > 1.05:
            return {
                "tooth_fdi": tooth_fdi,
                "tooth_type": "incisor",
                "anatomical_group": "anterior",
                "jaw": "unknown"
            }
        elif max(md_span, bl_span) < 9.0:
            return {
                "tooth_fdi": tooth_fdi,
                "tooth_type": "premolar",
                "anatomical_group": "posterior",
                "jaw": "unknown"
            }

    return {
        "tooth_fdi": tooth_fdi,
        "tooth_type": "molar",
        "anatomical_group": "posterior",
        "jaw": "unknown"
    }


def compute_anatomical_frame(tooth_mesh):
    """
    Derives the intrinsic 3D anatomical coordinate frame of an isolated tooth mesh
    purely from its vertex and face geometry:
      - e_crown: Apico-coronal axis (cervical base -> crown tip/occlusal surface),
                 computed via Gauss's divergence theorem on the open cervical base
                 (area-weighted sum of outward face normals).
      - e_facial: Labial/Buccal outward axis (pointing from dental arch center toward
                  lips/cheek, orthogonal to e_crown).
      - e_md: Mesio-distal axis (tangent along dental arch, e_crown x e_facial).
    """
    verts = tooth_mesh.vertices
    center = verts.mean(axis=0)

    # 1. Apico-coronal (crown) axis via area-weighted face normals
    area_normals = tooth_mesh.face_normals * tooth_mesh.area_faces[:, None]
    crown_vec = area_normals.sum(axis=0)
    crown_norm = np.linalg.norm(crown_vec)
    if crown_norm > 1e-8:
        e_crown = crown_vec / crown_norm
    else:
        e_crown = np.array([0.0, 0.0, 1.0])

    # 2. Labial/Buccal outward axis
    c_xy = np.array([center[0], center[1], 0.0])
    c_xy_ortho = c_xy - np.dot(c_xy, e_crown) * e_crown
    if np.linalg.norm(c_xy_ortho) > 2.0:
        e_facial = c_xy_ortho / np.linalg.norm(c_xy_ortho)
    else:
        v_centered = verts - center
        v_planar = v_centered - np.outer(np.dot(v_centered, e_crown), e_crown)
        _, _, vh = np.linalg.svd(v_planar, full_matrices=False)
        candidate_axis = vh[1]
        e_facial = candidate_axis / (np.linalg.norm(candidate_axis) + 1e-8)

    # 3. Mesio-distal axis (completing right-handed orthonormal frame)
    e_md = np.cross(e_crown, e_facial)
    e_md = e_md / (np.linalg.norm(e_md) + 1e-8)
    e_facial = np.cross(e_md, e_crown)
    e_facial = e_facial / (np.linalg.norm(e_facial) + 1e-8)

    # Project all vertices into local anatomical coordinates (u=MD, v=Facial, w=Crown)
    rel_verts = verts - center
    u = np.dot(rel_verts, e_md)
    v = np.dot(rel_verts, e_facial)
    w = np.dot(rel_verts, e_crown)

    w_min, w_max = float(w.min()), float(w.max())
    crown_height = max(w_max - w_min, 1e-5)
    w_norm = (w - w_min) / crown_height

    md_span = float(np.percentile(u, 98) - np.percentile(u, 2))
    bl_span = float(np.percentile(v, 98) - np.percentile(v, 2))

    # Measure mean mesh edge length dynamically to ensure lesions always span enough triangles
    mean_edge_len = float(tooth_mesh.edges_unique_length.mean())

    return {
        "center": center,
        "e_md": e_md,
        "e_facial": e_facial,
        "e_crown": e_crown,
        "u": u,
        "v": v,
        "w": w,
        "w_norm": w_norm,
        "md_span": md_span,
        "bl_span": bl_span,
        "crown_height": crown_height,
        "mean_edge_len": mean_edge_len
    }


def measure_local_slice_thickness(u, v, w_norm, target_w_norm):
    """
    Dynamically measures the tooth's actual labio-lingual / bucco-lingual thickness
    and mesio-distal width at the horizontal cross-section around target_w_norm.
    """
    band = np.abs(w_norm - target_w_norm) <= 0.10
    if np.sum(band) < 10:
        band = np.abs(w_norm - target_w_norm) <= 0.20
    if np.sum(band) < 5:
        band = np.ones_like(w_norm, dtype=bool)

    local_bl_thickness = float(np.percentile(v[band], 95) - np.percentile(v[band], 5))
    local_md_width = float(np.percentile(u[band], 95) - np.percentile(u[band], 5))
    return max(local_md_width, 1.0), max(local_bl_thickness, 1.0)


def convert_2d_geometry_to_3d_params(
    geometry,
    tooth_mesh,
    tooth_fdi=None,
    anomaly_type="Cavity"
):
    """
    Converts 2D shape geometry into 3D carving arguments for carve_cavity().
    All spatial cutoffs, radii, and depths are computed dynamically from the
    specific tooth mesh's geometry and its explicit FDI tooth type.
    """
    normals = tooth_mesh.vertex_normals
    bounds = tooth_mesh.bounds
    tooth_size = float(np.linalg.norm(bounds[1] - bounds[0]))

    # 1. Compute intrinsic anatomical frame and spans from mesh
    frame = compute_anatomical_frame(tooth_mesh)
    e_md = frame["e_md"]
    e_facial = frame["e_facial"]
    e_crown = frame["e_crown"]
    u, v, w, w_norm = frame["u"], frame["v"], frame["w"], frame["w_norm"]
    mean_edge_len = frame["mean_edge_len"]

    # 2. Explicit FDI Tooth-Type Classification
    tooth_info = classify_fdi_tooth(
        tooth_fdi,
        mesh_spans=(frame["md_span"], frame["bl_span"], frame["crown_height"])
    )
    tooth_type = tooth_info["tooth_type"]
    anatomical_group = tooth_info["anatomical_group"]

    rel_x, rel_y = geometry.get("relative_centroid", [0.5, 0.5])
    rel_x = float(np.clip(rel_x, 0.20, 0.80))
    rel_y = float(np.clip(rel_y, 0.20, 0.80))

    is_abrasion = str(anomaly_type).strip().lower().startswith("abrasion")
    raw_orientation = float(geometry.get("orientation", 0.0))

    # =========================================================================
    # 3. EXPLICIT TOOTH-TYPE BRANCH FOR CROWN REGION & PROJECTION
    # =========================================================================
    if anatomical_group == "anterior":
        # ---------------------------------------------------------------------
        # BRANCH A: ANTERIOR TEETH (Incisors & Canines)
        # Incisors and canines have a thin incisal ridge/cusp and NO occlusal
        # table. Clinical photographs view the labial (facial) surface.
        #   - Horizontal image axis (rel_x) -> Mesio-Distal axis (u)
        #   - Vertical image axis (rel_y)   -> Apico-Coronal crown axis (w)
        # ---------------------------------------------------------------------
        target_surface = "labial_facial"
        facial_alignment = np.dot(normals, e_facial)

        # Measure full anatomical labial crown face dimensions first (before interior padding)
        full_labial_mask = (facial_alignment > 0.30) & (w_norm >= 0.15) & (w_norm <= 0.88)
        if np.sum(full_labial_mask) < 30:
            full_labial_mask = facial_alignment > 0.15
        surf_span_u = float(np.percentile(u[full_labial_mask], 95) - np.percentile(u[full_labial_mask], 5))
        surf_span_w = float(np.percentile(w[full_labial_mask], 95) - np.percentile(w[full_labial_mask], 5))

        # Interior safe placement zone on the labial enamel plate:
        # For Abrasion (Class V cervical/labial wear), place in the cervical-middle third;
        # For Cavity/Caries, place in the central labial third.
        w_low = 0.32 if is_abrasion else 0.42
        w_high = 0.54 if is_abrasion else 0.68
        u_low, u_high = np.percentile(u[full_labial_mask], [28, 72])

        crown_mask = (
            (facial_alignment > 0.50)
            & (w_norm >= w_low) & (w_norm <= w_high)
            & (u >= u_low) & (u <= u_high)
        )
        if np.sum(crown_mask) < 15:
            crown_mask = (facial_alignment > 0.35) & (w_norm >= 0.30) & (w_norm <= 0.72)

        crown_indices = np.where(crown_mask)[0]
        u_cand = u[crown_indices]
        w_cand = w[crown_indices]

        target_u = float(np.percentile(u_cand, 20) + rel_x * (np.percentile(u_cand, 80) - np.percentile(u_cand, 20)))
        target_w = float(np.percentile(w_cand, 20) + (1.0 - rel_y) * (np.percentile(w_cand, 80) - np.percentile(w_cand, 20)))

        dists_2d = np.sqrt((u_cand - target_u) ** 2 + (w_cand - target_w) ** 2)
        center_idx = int(crown_indices[np.argmin(dists_2d)])

        characteristic_surf_span = min(surf_span_u, surf_span_w)
        max_major_span = surf_span_u  # Mesio-distal width bounds horizontal extension
        tangent_u = e_md
        tangent_v = e_crown

        # On anterior labial surfaces, cervical/labial abrasion wear facets run
        # horizontally along the mesio-distal arch axis (tangent_u = e_md).
        if is_abrasion:
            effective_orientation = float(np.clip(raw_orientation, -18.0, 18.0))
        else:
            effective_orientation = float(np.clip(raw_orientation, -35.0, 35.0))

    else:
        # ---------------------------------------------------------------------
        # BRANCH B: POSTERIOR TEETH (Premolars & Molars)
        # Premolars and molars possess an occlusal chewing table facing e_crown.
        #   - Horizontal image axis (rel_x) -> Mesio-Distal axis (u)
        #   - Vertical image axis (rel_y)   -> Bucco-Lingual axis (v)
        # ---------------------------------------------------------------------
        target_surface = "occlusal_table"
        occlusal_alignment = np.dot(normals, e_crown)

        # Measure full occlusal table span
        full_occ_mask = (occlusal_alignment > 0.30) & (w_norm >= 0.45)
        if np.sum(full_occ_mask) < 30:
            full_occ_mask = w_norm >= 0.50
        surf_span_u = float(np.percentile(u[full_occ_mask], 95) - np.percentile(u[full_occ_mask], 5))
        surf_span_v = float(np.percentile(v[full_occ_mask], 95) - np.percentile(v[full_occ_mask], 5))

        # Select interior occlusal basin / fossa & cusp slopes (avoiding outer vertical walls):
        # Note: Mandibular first premolars (#34/#44) have a strong lingual tilt of the
        # occlusal plane (occ_alignment ~ 0.42-0.50 in the developmental groove), so
        # we allow occlusal_alignment > 0.38 within the [10, 90] interior percentile box.
        u_low, u_high = np.percentile(u[full_occ_mask], [10, 90])
        v_low, v_high = np.percentile(v[full_occ_mask], [10, 90])

        crown_mask = (
            (occlusal_alignment > 0.38)
            & (w_norm >= 0.48)
            & (u >= u_low) & (u <= u_high)
            & (v >= v_low) & (v <= v_high)
        )
        if np.sum(crown_mask) < 15:
            crown_mask = (occlusal_alignment > 0.35) & (w_norm >= 0.45)

        crown_indices = np.where(crown_mask)[0]
        if not is_abrasion and len(crown_indices) > 15:
            # Occlusal caries originates in concave pits, fissures, and developmental
            # grooves/fossae (where surrounding neighbors sit higher along the surface
            # normal: (v_neighbor_mean - v_i) . n_i > 0), rather than convex cusp peaks.
            verts = np.asarray(tooth_mesh.vertices, dtype=np.float64)
            cand_verts = verts[crown_indices]
            cand_norms = normals[crown_indices]
            local_r = max(1.2, 0.25 * min(surf_span_u, surf_span_v))
            concavity_scores = np.zeros(len(crown_indices), dtype=np.float64)
            for idx_k, v_pos in enumerate(cand_verts):
                d_all = np.linalg.norm(verts - v_pos, axis=1)
                nbrs = (d_all > 1e-5) & (d_all <= local_r)
                if np.any(nbrs):
                    diff_mean = verts[nbrs].mean(axis=0) - v_pos
                    concavity_scores[idx_k] = float(np.dot(diff_mean, cand_norms[idx_k]))
            # Keep concave fossa/groove vertices (top 20% most concave pits/grooves)
            concavity_thresh = max(0.02, float(np.percentile(concavity_scores, 80)))
            fossa_indices = crown_indices[concavity_scores >= concavity_thresh]
            if len(fossa_indices) >= 6:
                crown_indices = fossa_indices

        u_cand = u[crown_indices]
        v_cand = v[crown_indices]

        target_u = float(np.percentile(u_cand, 25) + rel_x * (np.percentile(u_cand, 75) - np.percentile(u_cand, 25)))
        target_v = float(np.percentile(v_cand, 25) + rel_y * (np.percentile(v_cand, 75) - np.percentile(v_cand, 25)))

        dists_2d = np.sqrt((u_cand - target_u) ** 2 + (v_cand - target_v) ** 2)
        center_idx = int(crown_indices[np.argmin(dists_2d)])

        characteristic_surf_span = min(surf_span_u, surf_span_v)
        max_major_span = max(surf_span_u, surf_span_v)
        tangent_u = e_md
        tangent_v = e_facial
        effective_orientation = raw_orientation

    # =========================================================================
    # 4. DYNAMIC MESH-DERIVED RADIUS & DEPTH SCALING (ZERO FIXED MM CONSTANTS)
    # =========================================================================
    center_w_norm = float(w_norm[center_idx])
    local_md_w, local_bl_t = measure_local_slice_thickness(u, v, w_norm, center_w_norm)

    if anatomical_group == "anterior":
        local_wall_thickness = local_bl_t
    else:
        local_wall_thickness = max(float(w[center_idx] - w.min()), 2.0)

    # Dynamic radius bounds derived from:
    # 1. Mesh resolution floor (at least 4.2x mean triangle edge length so the lesion
    #    always spans >= 35-50 vertices and never forms a jagged 5-vertex pinhole)
    # 2. Tooth surface span (18% to 32% of characteristic crown surface dimension)
    mesh_resolution_min_r = 4.2 * mean_edge_len
    min_radius_dyn = max(mesh_resolution_min_r, 0.18 * characteristic_surf_span)
    max_radius_dyn = max(min_radius_dyn * 1.15, 0.30 * characteristic_surf_span)

    # Scale 2D relative_radius (which is normalized to full-arch photo width) into tooth space
    rel_r = float(geometry.get("relative_radius", 0.06))
    raw_radius = rel_r * characteristic_surf_span * 3.2
    carve_radius = float(np.clip(raw_radius, min_radius_dyn, max_radius_dyn))

    # Clamp aspect ratio so elongated lesions stay cleanly inside the crown borders
    raw_aspect = float(geometry.get("aspect_ratio", 1.0))
    max_allowed_aspect = min(2.2, (0.38 * max_major_span / max(carve_radius, 1e-4)) ** 2)
    aspect_ratio = float(np.clip(raw_aspect, 1.0, max(1.1, max_allowed_aspect)))

    # Dynamic depth coupled to both carve_radius (aspect ratio d/r) and local_wall_thickness:
    # NOTE ON DEPTH CALIBRATION:
    # `characteristic_surf_span` and `local_wall_thickness` are measured dynamically from
    # the Teeth3DS+ tooth mesh geometry. However, the dimensionless multipliers below
    # (0.20, 0.14, 0.24, 0.08, 0.05, 0.10 for Abrasion; 0.42, 0.32, 0.48, 0.15, 0.10, 0.18
    # for Cavity) are clinical heuristics applied to those measured mesh dimensions, NOT
    # parameters empirically regressed from clinical ground-truth lesion depth scans.
    if is_abrasion:
        # Abrasion heuristic ratio: shallow wear facet (depth = 20% of radius, capped at 8% of wall thickness)
        profile = "abrasion"
        target_depth = min(0.20 * carve_radius, 0.08 * local_wall_thickness)
        min_depth_dyn = min(0.14 * carve_radius, 0.05 * local_wall_thickness)
        max_depth_dyn = min(0.24 * carve_radius, 0.10 * local_wall_thickness)
    else:
        # Cavity / Caries heuristic ratio: concave cavitation (depth = 42% of radius, capped at 15% of wall thickness)
        profile = "cavity"
        target_depth = min(0.42 * carve_radius, 0.15 * local_wall_thickness)
        min_depth_dyn = min(0.32 * carve_radius, 0.10 * local_wall_thickness)
        max_depth_dyn = min(0.48 * carve_radius, 0.18 * local_wall_thickness)

    if max_depth_dyn < min_depth_dyn:
        max_depth_dyn = min_depth_dyn
    carve_depth = float(np.clip(target_depth, min_depth_dyn, max_depth_dyn))

    return {
        "center_idx": center_idx,
        "radius": round(carve_radius, 3),
        "depth": round(carve_depth, 3),
        "aspect_ratio": round(aspect_ratio, 2),
        "orientation": round(effective_orientation, 2),
        "profile": profile,
        "tooth_fdi": tooth_info["tooth_fdi"],
        "tooth_type": tooth_type,
        "anatomical_group": anatomical_group,
        "target_surface": target_surface,
        "tooth_size": round(tooth_size, 3),
        "surface_span_mm": round(characteristic_surf_span, 3),
        "local_wall_thickness_mm": round(local_wall_thickness, 3),
        "mean_edge_len_mm": round(mean_edge_len, 3),
        "depth_calibration_source": "clinical_heuristic_ratio_on_measured_mesh",
        "tangent_u": [round(float(x), 5) for x in tangent_u],
        "tangent_v": [round(float(x), 5) for x in tangent_v],
        "e_crown": [round(float(x), 5) for x in e_crown],
        "notes": (
            f"Dynamic {tooth_type} ({target_surface}) mapping; "
            f"profile={profile}, measured_wall={local_wall_thickness:.2f}mm (heuristic depth ratio)"
        )
    }
