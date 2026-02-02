#!/usr/bin/env python3
"""
tool3d_print_v1_4_1_final.py

3D Printability Analysis + Blueprint Export (SVG / DXF)
Scope: Blueprint-Annotation (K3), keine Geometrie-Mutation
Normen: ISO 128-1:2020, ISO 7200
Status: FREEZE - nur Bugfix-Patches
"""

import sys
import numpy as np
import trimesh
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import ezdxf
import svgwrite
from svgwrite import mm

# ------------------------------------------------------------
# INDUSTRY THRESHOLDS (2025)
# ------------------------------------------------------------

THRESHOLDS = {
    "min_wall_mm": 0.8,
    "recommended_wall_mm": 1.2,
    "max_overhang_A": 45.0,
    "max_overhang_B": 60.0,
    "min_feature_mm": 0.6,
}

# ------------------------------------------------------------
# DATA MODEL (ISO 7200 compliant)
# ------------------------------------------------------------


@dataclass
class BlueprintAnnotation:
    score: str
    note: str
    details: dict
    warnings: list
    position: tuple

    # ISO 7200 mandatory
    title: str = "3D Printability Analysis"
    drawing_number: str = ""
    date_of_issue: str = ""
    status: str = "Draft"
    projection_method: str = "ISO-E (First Angle)"
    scale: str = "1:1"

    # Optional / best practice
    legal_owner: str = ""
    language: str = "en"
    units: str = "mm"

    # Internal
    timestamp: str = ""
    k3_version: str = "1.4.1"


# ------------------------------------------------------------
# CORE ANALYSIS
# ------------------------------------------------------------


def _ray_thickness(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """
    Berechnet die Wandstaerke pro Punkt ueber Bi-Directional Ray-Casting.

    Prinzip: Fuer jeden Punkt wird in +Z und -Z geschossen.
    Die Summe der beiden naechsten Trefferabstaende ~ lokale Wandstaerke.
    Gibt NaN zurueck wenn kein gueltiger Treffer existiert.
    """
    thickness = np.full(len(points), np.nan)

    for i, pt in enumerate(points):
        hits_pos = mesh.ray.intersects_location(
            ray_origins=[pt],
            ray_directions=[[0, 0, 1]],
        )[0]
        hits_neg = mesh.ray.intersects_location(
            ray_origins=[pt],
            ray_directions=[[0, 0, -1]],
        )[0]

        if len(hits_pos) == 0 or len(hits_neg) == 0:
            continue

        # Naechsten Treffer in jeweiliger Richtung
        d_pos = np.linalg.norm(hits_pos - pt, axis=1).min()
        d_neg = np.linalg.norm(hits_neg - pt, axis=1).min()
        thickness[i] = d_pos + d_neg

    return thickness


def measure_wall_thickness(mesh: trimesh.Trimesh) -> dict:
    """
    Wandstaerke-Messung.

    Strategie:
      1. Watertight-Pruefung -> sonst Fallback auf Bounding-Box-Extrapolation
      2. Ray-Casting ueber Flaechenzentren (Sampling capped bei 5000)
      3. NaN-Filterung vor Auswertung
    """
    if not mesh.is_watertight:
        return {
            "status": "FAIL",
            "t_min": float(mesh.extents.min()),
            "position": tuple(mesh.centroid),
            "method": "bbox_fallback",
            "error": "mesh_not_watertight",
        }

    face_centers = mesh.triangles.mean(axis=1)

    # Sampling cappen bei grossen Meshes fuer Performance
    max_samples = 5000
    if len(face_centers) > max_samples:
        indices = np.random.default_rng(42).choice(
            len(face_centers), size=max_samples, replace=False
        )
        face_centers = face_centers[indices]

    try:
        thickness = _ray_thickness(mesh, face_centers)

        valid_mask = ~np.isnan(thickness)
        if not valid_mask.any():
            raise ValueError("no_valid_ray_hits")

        valid_thickness = thickness[valid_mask]
        valid_centers = face_centers[valid_mask]
        idx = int(np.argmin(valid_thickness))

        return {
            "status": "PASS",
            "t_min": float(valid_thickness[idx]),
            "position": tuple(valid_centers[idx]),
            "method": "ray",
            "samples": int(valid_mask.sum()),
        }

    except Exception as e:
        return {
            "status": "FAIL",
            "t_min": float(mesh.extents.min()),
            "position": tuple(mesh.centroid),
            "method": "bbox_fallback",
            "error": str(e),
        }


def max_overhang_angle(mesh: trimesh.Trimesh) -> float:
    """
    Maximaler Ueberhangwinkel relativ zur Druckrichtung (+Z).

    Berechnung:
      - Winkel zwischen Face-Normal und +Z via arccos(dot)
      - Ueberhang = Winkel > 90 deg (Normal zeigt nach unten)
      - Rueckgabe: Maximaler Winkel in Grad [0 deg..180 deg]
        Praktisch relevanter Bereich: 90 deg..180 deg = Ueberhang
        Fuer Vergleich mit Thresholds wird auf (angle - 90 deg) reduziert,
        damit 0 deg = horizontal, 90 deg = vollstaendig nach unten.
    """
    normals = mesh.face_normals
    z_up = np.array([0.0, 0.0, 1.0])

    # Winkel zwischen Normal und +Z: 0 deg = nach oben, 180 deg = nach unten
    cos_angles = np.clip(normals @ z_up, -1.0, 1.0)
    angles_from_z = np.degrees(np.arccos(cos_angles))

    # Ueberhangwinkel = wie weit ueber die Horizontale (90 deg) hinaus
    # 0 deg = horizontal, positiv = Ueberhang nach unten
    overhang_angles = angles_from_z - 90.0

    # Nur echte Ueberhaenge (> 0 deg) relevanz
    max_overhang = float(np.max(overhang_angles))
    return max(max_overhang, 0.0)


def _adaptive_eps(points: np.ndarray, k: int = 4) -> float:
    """
    Adaptive eps-Bestimmung fuer DBSCAN via K-Nearest-Neighbor Elbow-Methode.

    Algorithmus:
      1. K-Distanzen berechnen und sortieren
      2. Zweite Ableitung (Kruemmung) bestimmen
      3. Elbow = Maximum der Kruemmung (schaerfster Knick)
      4. Kein kuenstlicher Clamp - Elbow wird so wie gefunden verwendet
    """
    if len(points) < k:
        return 5.0

    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    k_dist = np.sort(distances[:, -1])

    if len(k_dist) < 10:
        return float(np.percentile(k_dist, 90))

    # Zweite Ableitung der sortierten K-Distanzen
    curvature = np.gradient(np.gradient(k_dist))

    # Elbow = schaerfster Anstieg der Kruemmung
    elbow = int(np.argmax(curvature))

    return float(k_dist[elbow])


def find_overhang_position(mesh: trimesh.Trimesh, threshold: float = 45.0) -> tuple:
    """
    Findet die Position des dominanten Ueberhangbereichs.

    Strategie:
      - Bei kleinen Meshes oder wenigen Ueberhangflaechen: direkt argmax
      - Bei grossen Meshes: DBSCAN-Clustering -> groesster Cluster -> Centroid
    """
    normals = mesh.face_normals
    z_up = np.array([0.0, 0.0, 1.0])

    cos_angles = np.clip(normals @ z_up, -1.0, 1.0)
    angles_from_z = np.degrees(np.arccos(cos_angles))
    overhang_angles = angles_from_z - 90.0

    mask = overhang_angles > threshold
    if not mask.any():
        # Kein Ueberhang ueber Threshold -> Centroid zurueckgeben
        return tuple(mesh.centroid)

    overhang_centers = mesh.triangles[mask].mean(axis=1)
    overhang_vals = overhang_angles[mask]

    # Kleines Mesh oder wenige Punkte -> direkt maximalen Punkt
    if len(mesh.faces) < 50_000 or len(overhang_centers) < 3:
        idx = int(np.argmax(overhang_vals))
        return tuple(overhang_centers[idx])

    # Grosses Mesh -> DBSCAN fuer dominanten Cluster
    eps = _adaptive_eps(overhang_centers)
    clustering = DBSCAN(eps=eps, min_samples=3, algorithm="kd_tree").fit(overhang_centers)
    labels = clustering.labels_

    valid_labels = labels[labels != -1]
    if len(valid_labels) == 0:
        # Alle Punkte sind Outlier -> Punkt mit max Ueberhang
        idx = int(np.argmax(overhang_vals))
        return tuple(overhang_centers[idx])

    # Groesster Cluster -> Centroid
    dominant_label = max(set(valid_labels), key=lambda lbl: (labels == lbl).sum())
    cluster_mask = labels == dominant_label
    return tuple(overhang_centers[cluster_mask].mean(axis=0))


def printability_score(mesh: trimesh.Trimesh) -> dict:
    """
    Hauptbewertung: Wandstaerke + Ueberhang -> Score A/B/C.

    Scoring-Logik:
      A = Direkt druckbar (Wand OK, Ueberhang <= 45 deg)
      B = Supports empfohlen (Wand OK, Ueberhang 45 deg-60 deg)
      C = Supports erforderlich ODER Wandstaerke kritisch

    Hinweis: Beide Metriken werden IMMER berechnet, auch bei C,
    damit der Report vollstaendig und audit-faehig ist.
    """
    warnings = []

    wall = measure_wall_thickness(mesh)
    overhang = max_overhang_angle(mesh)

    details = {
        "wall_thickness_mm": wall,
        "max_overhang_deg": round(overhang, 1),
    }

    # Wandstaerke-Warnung (blockiert nicht mehr allein)
    wall_critical = False
    if wall["t_min"] < THRESHOLDS["min_wall_mm"]:
        warnings.append(
            f"Wall thickness {wall['t_min']:.2f} mm < {THRESHOLDS['min_wall_mm']} mm (critical)"
        )
        wall_critical = True
    elif wall["t_min"] < THRESHOLDS["recommended_wall_mm"]:
        warnings.append(
            f"Wall thickness {wall['t_min']:.2f} mm below recommended "
            f"{THRESHOLDS['recommended_wall_mm']} mm"
        )

    # Ueberhang-Bewertung
    if overhang > THRESHOLDS["max_overhang_B"]:
        warnings.append(f"Overhang {overhang:.1f} deg requires supports")
        return _result("C", "Supports required", details, warnings)

    if overhang > THRESHOLDS["max_overhang_A"]:
        warnings.append(f"Overhang {overhang:.1f} deg may require supports")
        if wall_critical:
            return _result("C", "Wall thickness below minimum + supports needed", details, warnings)
        return _result("B", "Supports recommended", details, warnings)

    # Ueberhang OK - nur Wall entscheidet
    if wall_critical:
        return _result("C", "Wall thickness below minimum", details, warnings)

    return _result("A", "Directly printable without supports", details, warnings)


def _result(score: str, note: str, details: dict, warnings: list) -> dict:
    return {
        "score": score,
        "note": note,
        "details": details,
        "warnings": warnings,
    }


# ------------------------------------------------------------
# BLUEPRINT BUILD
# ------------------------------------------------------------


def build_blueprint(
    mesh: trimesh.Trimesh,
    result: dict,
    drawing_number: str = "AUTO-001",
) -> BlueprintAnnotation:
    """
    Erstellt eine ISO 7200-konforme BlueprintAnnotation aus dem Analyse-Ergebnis.

    Position: offset vom Mesh-Bounding-Box-Minimum,
    aber auf mindestens (0, 0) geclampt damit SVG korrekt rendert.
    """
    bounds = mesh.bounds
    now = datetime.now(timezone.utc)

    # Position clampen auf nicht-negative Werte (SVG-Kompatibilitaet)
    pos_x = max(0.0, float(bounds[0][0]) - 5.0)
    pos_y = max(0.0, float(bounds[0][1]) - 5.0)
    pos_z = float(bounds[0][2])

    return BlueprintAnnotation(
        score=result["score"],
        note=result["note"],
        details=result["details"],
        warnings=result["warnings"],
        position=(pos_x, pos_y, pos_z),
        drawing_number=drawing_number,
        date_of_issue=now.strftime("%Y-%m-%d"),
        timestamp=now.isoformat(),
    )


# ------------------------------------------------------------
# EXPORTS
# ------------------------------------------------------------


def export_svg(annotation: BlueprintAnnotation, out_path: str) -> None:
    """SVG-Export: Blueprint-Annotation mit Score, Notiz und Metadaten."""
    dwg = svgwrite.Drawing(out_path, profile="full")
    x, y, _ = annotation.position

    # Score - Hauptzeile
    dwg.add(dwg.text(
        f"PRINTABILITY SCORE: {annotation.score}",
        insert=(x * mm, y * mm),
        font_size="5mm",
        font_weight="bold",
        font_family="Arial, sans-serif",
    ))

    # Beschreibung
    dwg.add(dwg.text(
        annotation.note,
        insert=(x * mm, (y + 8) * mm),
        font_size="4mm",
        font_family="Arial, sans-serif",
    ))

    # Metadaten-Zeile
    dwg.add(dwg.text(
        f"Drawing: {annotation.drawing_number} | {annotation.date_of_issue}",
        insert=(x * mm, (y + 16) * mm),
        font_size="2.5mm",
        fill="gray",
        font_family="Arial, sans-serif",
    ))

    # Warnings (falls vorhanden)
    for i, warning in enumerate(annotation.warnings):
        dwg.add(dwg.text(
            f"! {warning}",
            insert=(x * mm, (y + 22 + i * 5) * mm),
            font_size="2.5mm",
            fill="#cc0000",
            font_family="Arial, sans-serif",
        ))

    dwg.save()


def export_dxf(annotation: BlueprintAnnotation, out_path: str) -> None:
    """
    DXF R2018 Export mit korrektem Annotative-Handling.

    Layer-Struktur:
      K3_TEXT      - Haupttext (Score)
      K3_WARNINGS  - Warnungen
      K3_META      - Metadaten (Drawing Number, Datum)

    Annotative: wird ueber DXF-Attrib 'annotative' gesetzt (nicht per Style-Methode).
    """
    doc = ezdxf.new("R2018", setup=True)
    msp = doc.modelspace()

    # Layer-Definitionen
    doc.layers.add("K3_TEXT", color=2)
    doc.layers.add("K3_WARNINGS", color=1)
    doc.layers.add("K3_META", color=8)

    x, y, _ = annotation.position

    # Haupttext - Score
    msp.add_text(
        f"PRINTABILITY: {annotation.score}",
        dxfattribs={
            "layer": "K3_TEXT",
            "height": 5.0,
            "insert": (x, y),
            "annotative": True,
        },
    )

    # Beschreibung
    msp.add_text(
        annotation.note,
        dxfattribs={
            "layer": "K3_TEXT",
            "height": 3.5,
            "insert": (x, y - 7),
            "annotative": True,
        },
    )

    # Metadaten
    msp.add_text(
        f"Drawing {annotation.drawing_number} | {annotation.date_of_issue}",
        dxfattribs={
            "layer": "K3_META",
            "height": 2.5,
            "insert": (x, y - 12),
        },
    )

    # Warnings
    for i, warning in enumerate(annotation.warnings):
        msp.add_text(
            f"WARNING: {warning}",
            dxfattribs={
                "layer": "K3_WARNINGS",
                "height": 2.0,
                "insert": (x, y - 18 - i * 3.5),
            },
        )

    doc.saveas(out_path)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: tool3d_print_v1_4_1_final.py <mesh.stl> [--drawing-number <NUM>]")
        print()
        print("Options:")
        print("  --drawing-number <NUM>   DXF Drawing Number (default: AUTO-001)")
        sys.exit(1)

    # Argument-Parsing
    mesh_path = Path(sys.argv[1])
    drawing_number = "AUTO-001"

    if "--drawing-number" in sys.argv:
        try:
            idx = sys.argv.index("--drawing-number")
            drawing_number = sys.argv[idx + 1]
        except IndexError:
            print("Error: --drawing-number requires a value")
            sys.exit(1)

    # Validierung
    if not mesh_path.exists():
        print(f"Error: file not found: {mesh_path}")
        sys.exit(1)

    if mesh_path.suffix.lower() not in (".stl", ".obj", ".off", ".ply", ".glb", ".gltf"):
        print(f"Error: unsupported file format: {mesh_path.suffix}")
        sys.exit(1)

    # Mesh laden
    try:
        mesh = trimesh.load(str(mesh_path), force="mesh")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        sys.exit(1)

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        print("Error: invalid or empty mesh")
        sys.exit(1)

    # Analyse
    result = printability_score(mesh)
    blueprint = build_blueprint(mesh, result, drawing_number=drawing_number)

    # Export
    svg_path = "blueprint.svg"
    dxf_path = "blueprint.dxf"

    export_svg(blueprint, svg_path)
    export_dxf(blueprint, dxf_path)

    # Ausgabe
    print("Analysis complete")
    print(f"  Score:   {result['score']} - {result['note']}")
    print(f"  Outputs: {svg_path}, {dxf_path}")

    if result["warnings"]:
        print()
        print("  Warnings:")
        for w in result["warnings"]:
            print(f"    ! {w}")


if __name__ == "__main__":
    main()
