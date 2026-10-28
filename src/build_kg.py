"""
build_kg.py  –  Manufacturing Engineering Change Impact Knowledge Graph Builder
==============================================================================
Usage:
    python src/build_kg.py          # idempotent load / update
    python src/build_kg.py --wipe   # delete everything first, then load

Reads credentials from .env  (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).

NOTE: GT_IMPACTS relationships are the ANSWER KEY for evaluation / GNN training.
      Agent traversal queries must NEVER traverse GT_IMPACTS edges.

Changes vs. previous version
----------------------------
1. Added QualityRequirement node label (+ uniqueness constraint).
2. Added (Component)-[:MUST_MEET]->(QualityRequirement).
3. Added (ECO)-[:SIMILAR_TO]->(ECO)  (rule-based; does NOT use GT_IMPACTS).
4. Added spec-named relationships alongside the existing ones:
     (Component)-[:PART_OF]->(Assembly)     (Subsystem nodes also get :Assembly label)
     (Process)-[:REQUIRES]->(Machine)
     (Component)-[:SUPPLIED_BY]->(Supplier)
5. BUG FIX: edge properties from graph_edges.csv (cpk, lead_time_days, quantity, ...)
   are now actually stored on the relationships (previously parsed but never SET).
"""

import argparse
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

# ── env ────────────────────────────────────────────────────────────────────────
load_dotenv(ROOT / ".env")
URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme123")

BATCH = 1000   # UNWIND batch size


# ══════════════════════════════════════════════════════════════════════════════
# CSV HELPERS  (data-quirk repairs)
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(filename: str, **kwargs) -> pd.DataFrame:
    """Simple UTF-8-BOM aware CSV reader."""
    return pd.read_csv(DATA / filename, encoding="utf-8-sig", **kwargs)


def repair_csv_extra_fields(filename: str, expected_cols: int, merge_index: int) -> pd.DataFrame:
    """
    Read a CSV where a free-text column (merge_index) may contain unquoted
    commas, causing rows to have more fields than expected_cols.
    Surplus fields are merged back into the column at merge_index.
    """
    raw = (DATA / filename).read_text(encoding="utf-8-sig")
    reader = csv.reader(io.StringIO(raw))
    rows = list(reader)
    header = rows[0]

    fixed = []
    for row in rows[1:]:
        if not row:
            continue
        surplus = len(row) - expected_cols
        if surplus > 0:
            merged_field = ",".join(row[merge_index: merge_index + surplus + 1])
            row = row[:merge_index] + [merged_field] + row[merge_index + surplus + 1:]
        fixed.append(row)

    return pd.DataFrame(fixed, columns=header)


def parse_json_properties(s: str) -> dict:
    """
    Robustly parse the mangled properties_json in graph_edges.csv.
    Falls back to regex extraction of key:number pairs.
    e.g. {\\quantity\":1}" → {"quantity": 1}
    """
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        parsed = json.loads(s)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        result = {}
        for key, val in re.findall(r'([A-Za-z_]+)\\*"?:\s*([\d.]+)', s):
            try:
                result[key] = float(val) if "." in val else int(val)
            except ValueError:
                pass
        return result


def coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe(val):
    """Convert NaN / NA to None so Neo4j accepts it."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def clean_props(props: dict) -> dict:
    """Keep only scalar (str/int/float/bool) edge properties — Neo4j-safe."""
    out = {}
    for k, v in (props or {}).items():
        if isinstance(v, (str, int, float, bool)):
            out[str(k)] = v
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_data() -> dict:
    print("Loading CSV files …")

    # -- components (16 cols; index 8 = cad_dimensions may have extra commas) --
    # Actual columns: component_id, component_name, subsystem, material_id, material_name,
    #   primary_process_id, nominal_mass_g, nominal_tolerance_mm, cad_dimensions,
    #   material_density_g_cm3, material_yield_strength_mpa, material_machinability_index,
    #   material_cost_index, governing_sop_id, design_revision, synthetic_assumption_flag
    components = repair_csv_extra_fields("components.csv", expected_cols=16, merge_index=8)
    coerce_numeric(components, ["nominal_mass_g", "nominal_tolerance_mm",
                                "material_density_g_cm3", "material_yield_strength_mpa",
                                "material_machinability_index", "material_cost_index"])

    # -- engineering_changes (13 cols; index 10 = technical_description) --------
    # Actual columns: change_id, change_family, target_node_id, target_node_type,
    #   change_type, current_value, proposed_value, change_intent, direct_severity,
    #   primary_impact_dimension, technical_description, approval_status, synthetic_data_flag
    eco = repair_csv_extra_fields("engineering_changes.csv", expected_cols=13, merge_index=10)
    eco.columns = eco.columns.str.strip()

    # -- simple reads -----------------------------------------------------------
    bom       = read_csv("bom_relationships.csv")
    materials = read_csv("materials.csv")
    processes = read_csv("processes.csv")
    machines  = read_csv("machines.csv")
    suppliers = read_csv("suppliers.csv")
    sops      = read_csv("sops.csv")
    labels    = read_csv("change_impact_labels.csv")

    # -- graph_edges: repair mangled JSON properties ---------------------------
    graph_edges = read_csv("graph_edges.csv")
    graph_edges["props"] = graph_edges["properties_json"].apply(parse_json_properties)

    # -- production_records: aggregate only (do NOT load as nodes) -------------
    print("  Aggregating production_records.csv (50 k rows) …")
    prod = pd.read_csv(DATA / "production_records.csv", encoding="utf-8-sig",
                       usecols=["component_id", "machine_id",
                                "defect_flag", "cycle_time_min",
                                "dimensional_deviation_mm"])
    coerce_numeric(prod, ["defect_flag", "cycle_time_min", "dimensional_deviation_mm"])

    prod_comp = (prod.groupby("component_id")
                     .agg(prod_ops_count=("defect_flag", "count"),
                          prod_defect_rate=("defect_flag", "mean"),
                          prod_cycle_time_min=("cycle_time_min", "mean"),
                          prod_dim_deviation_mm=("dimensional_deviation_mm", "mean"))
                     .reset_index())

    prod_mach = (prod.groupby("machine_id")
                     .agg(prod_ops_count=("defect_flag", "count"),
                          prod_defect_rate=("defect_flag", "mean"),
                          prod_cycle_time_min=("cycle_time_min", "mean"),
                          prod_dim_deviation_mm=("dimensional_deviation_mm", "mean"))
                     .reset_index())

    return dict(
        components=components, bom=bom, eco=eco,
        materials=materials, processes=processes, machines=machines,
        suppliers=suppliers, sops=sops, labels=labels,
        graph_edges=graph_edges,
        prod_comp=prod_comp, prod_mach=prod_mach,
    )


# ══════════════════════════════════════════════════════════════════════════════
# NEO4J HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_batched(session, query: str, records: list, batch: int = BATCH):
    """UNWIND a list in batches."""
    for i in range(0, len(records), batch):
        session.run(query, rows=records[i: i + batch])


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA CONSTRAINTS
# ══════════════════════════════════════════════════════════════════════════════

# Fixed vocabulary — no user input, safe from injection
LABELS = ["Product", "Subsystem", "Component", "Material",
          "Process", "Machine", "Supplier", "SOP", "ECO",
          "QualityRequirement"]


def create_constraints(driver):
    print("Creating uniqueness constraints …")
    with driver.session() as s:
        for label in LABELS:
            s.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) "
                f"REQUIRE n.id IS UNIQUE"
            )


# ══════════════════════════════════════════════════════════════════════════════
# NODE LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_product_subsystem_nodes(session, graph_edges: pd.DataFrame):
    """
    Infer Product and Subsystem nodes from graph_edges.
    source_node_type == 'PRODUCT' → Product node
    targets of Product-sourced edges → Subsystem nodes
    """
    prod_rows  = graph_edges[graph_edges["source_node_type"] == "PRODUCT"]

    products   = prod_rows["source_node_id"].unique()
    subsystems = prod_rows["target_node_id"].unique()  # targets of Product-CONTAINS edges

    print(f"  Loading {len(products)} Product node(s) …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (n:Product {id: r.id})
        SET n.name = r.id
    """, [{"id": p} for p in products])

    print(f"  Loading {len(subsystems)} Subsystem node(s) …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (n:Subsystem {id: r.id})
        SET n.name = r.id
    """, [{"id": s} for s in subsystems])


def load_component_nodes(session, components: pd.DataFrame, prod_comp: pd.DataFrame):
    """
    Components CSV cols:
      component_id, component_name, subsystem, material_id, material_name,
      primary_process_id, nominal_mass_g, nominal_tolerance_mm, cad_dimensions,
      material_density_g_cm3, material_yield_strength_mpa, material_machinability_index,
      material_cost_index, governing_sop_id, design_revision, synthetic_assumption_flag
    """
    prod_map = prod_comp.set_index("component_id").to_dict("index")
    records = []
    for _, row in components.iterrows():
        cid = safe(row["component_id"])
        if cid is None:
            continue
        props = {
            "id":              cid,
            "name":            safe(row.get("component_name")),
            "subsystem":       safe(row.get("subsystem")),
            "material_id":     safe(row.get("material_id")),
            "process_id":      safe(row.get("primary_process_id")),
            "governing_sop":   safe(row.get("governing_sop_id")),
            "nominal_mass_g":  safe(row.get("nominal_mass_g")),
            "nominal_tolerance_mm": safe(row.get("nominal_tolerance_mm")),
            "cad_dimensions":  safe(row.get("cad_dimensions")),
            "design_revision": safe(row.get("design_revision")),
        }
        if cid in prod_map:
            p = prod_map[cid]
            props.update({
                "prod_ops_count":        safe(p.get("prod_ops_count")),
                "prod_defect_rate":      safe(p.get("prod_defect_rate")),
                "prod_cycle_time_min":   safe(p.get("prod_cycle_time_min")),
                "prod_dim_deviation_mm": safe(p.get("prod_dim_deviation_mm")),
            })
        records.append(props)

    print(f"  Loading {len(records)} Component node(s) …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (n:Component {id: r.id})
        SET n += r
    """, records)


def load_material_nodes(session, materials: pd.DataFrame):
    """
    Materials CSV cols:
      material_id, material_name, material_class, density_g_cm3, yield_strength_mpa,
      ultimate_strength_mpa, thermal_conductivity_w_mk, machinability_index,
      raw_cost_index, co2_footprint_kg_per_kg
    """
    records = []
    for _, row in materials.iterrows():
        mid = safe(row.get("material_id"))
        if mid is None:
            continue
        records.append({
            "id":                   mid,
            "name":                 safe(row.get("material_name")),
            "category":             safe(row.get("material_class")),
            "density_g_cm3":        safe(row.get("density_g_cm3")),
            "yield_strength_mpa":   safe(row.get("yield_strength_mpa")),
            "ultimate_strength_mpa":safe(row.get("ultimate_strength_mpa")),
            "thermal_conductivity": safe(row.get("thermal_conductivity_w_mk")),
            "machinability_index":  safe(row.get("machinability_index")),
            "raw_cost_index":       safe(row.get("raw_cost_index")),
            "co2_kg_per_kg":        safe(row.get("co2_footprint_kg_per_kg")),
        })
    print(f"  Loading {len(records)} Material node(s) …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (n:Material {id: r.id})
        SET n += r
    """, records)


def load_process_nodes(session, processes: pd.DataFrame):
    """
    Processes CSV cols:
      process_id, process_name, machine_type, cycle_mean_min, cycle_sd_min,
      power_rating_kw, energy_mean_kwh, governing_sop_id
    """
    records = []
    for _, row in processes.iterrows():
        pid = safe(row.get("process_id"))
        if pid is None:
            continue
        records.append({
            "id":             pid,
            "name":           safe(row.get("process_name")),
            "machine_type":   safe(row.get("machine_type")),
            "cycle_mean_min": safe(row.get("cycle_mean_min")),
            "cycle_sd_min":   safe(row.get("cycle_sd_min")),
            "power_kw":       safe(row.get("power_rating_kw")),
            "energy_kwh":     safe(row.get("energy_mean_kwh")),
            "governing_sop":  safe(row.get("governing_sop_id")),
        })
    print(f"  Loading {len(records)} Process node(s) …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (n:Process {id: r.id})
        SET n += r
    """, records)


def load_machine_nodes(session, machines: pd.DataFrame, prod_mach: pd.DataFrame):
    """
    Machines CSV cols:
      machine_id, machine_name, compatible_process_id, machine_type, machine_age_years,
      maintenance_status, capability_index_cpk, energy_efficiency_factor, synthetic_assumption_flag
    """
    prod_map = prod_mach.set_index("machine_id").to_dict("index")
    records = []
    for _, row in machines.iterrows():
        mid = safe(row.get("machine_id"))
        if mid is None:
            continue
        props = {
            "id":                 mid,
            "name":               safe(row.get("machine_name")),
            "compatible_process": safe(row.get("compatible_process_id")),
            "machine_type":       safe(row.get("machine_type")),
            "age_years":          safe(row.get("machine_age_years")),
            "maintenance_status": safe(row.get("maintenance_status")),
            "cpk":                safe(row.get("capability_index_cpk")),
            "energy_efficiency":  safe(row.get("energy_efficiency_factor")),
        }
        if mid in prod_map:
            p = prod_map[mid]
            props.update({
                "prod_ops_count":        safe(p.get("prod_ops_count")),
                "prod_defect_rate":      safe(p.get("prod_defect_rate")),
                "prod_cycle_time_min":   safe(p.get("prod_cycle_time_min")),
                "prod_dim_deviation_mm": safe(p.get("prod_dim_deviation_mm")),
            })
        records.append(props)
    print(f"  Loading {len(records)} Machine node(s) …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (n:Machine {id: r.id})
        SET n += r
    """, records)


def load_supplier_nodes(session, suppliers: pd.DataFrame):
    """
    Suppliers CSV cols:
      supplier_id, supplier_name, supplier_tier, quality_rating, on_time_delivery_pct,
      lead_time_days, qualification_status, certified_materials, synthetic_assumption_flag
    """
    records = []
    for _, row in suppliers.iterrows():
        sid = safe(row.get("supplier_id"))
        if sid is None:
            continue
        cert_raw = safe(row.get("certified_materials", ""))
        certified = []
        if cert_raw:
            certified = [x.strip() for x in re.split(r"[;,]", str(cert_raw)) if x.strip()]
        records.append({
            "id":                   sid,
            "name":                 safe(row.get("supplier_name")),
            "tier":                 safe(row.get("supplier_tier")),
            "quality_rating":       safe(row.get("quality_rating")),
            "on_time_delivery_pct": safe(row.get("on_time_delivery_pct")),
            "lead_time_days":       safe(row.get("lead_time_days")),
            "qualification_status": safe(row.get("qualification_status")),
            "certified_materials":  certified,
        })
    print(f"  Loading {len(records)} Supplier node(s) …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (n:Supplier {id: r.id})
        SET n += r
    """, records)


def load_sop_nodes(session, sops: pd.DataFrame):
    """
    SOPs CSV cols: sop_id, title, revision, scope, critical_tolerances, risk_classification
    """
    records = []
    for _, row in sops.iterrows():
        sid = safe(row.get("sop_id"))
        if sid is None:
            continue
        records.append({
            "id":                  sid,
            "name":                safe(row.get("title")),
            "revision":            safe(row.get("revision")),
            "scope":               safe(row.get("scope")),
            "critical_tolerances": safe(row.get("critical_tolerances")),
            "risk":                safe(row.get("risk_classification")),
        })
    print(f"  Loading {len(records)} SOP node(s) …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (n:SOP {id: r.id})
        SET n += r
    """, records)


def load_eco_nodes(session, eco: pd.DataFrame):
    """
    ECO CSV cols:
      change_id, change_family, target_node_id, target_node_type, change_type,
      current_value, proposed_value, change_intent, direct_severity,
      primary_impact_dimension, technical_description, approval_status, synthetic_data_flag
    """
    records = []
    for _, row in eco.iterrows():
        eid = safe(row.get("change_id"))
        if eid is None:
            continue
        records.append({
            "id":               eid,
            "family":           safe(row.get("change_family")),
            "change_type":      safe(row.get("change_type")),
            "target_node_id":   safe(row.get("target_node_id")),
            "target_node_type": safe(row.get("target_node_type")),
            "current_value":    safe(row.get("current_value")),
            "proposed_value":   safe(row.get("proposed_value")),
            "intent":           safe(row.get("change_intent")),
            "direct_severity":  safe(row.get("direct_severity")),
            "impact_dimension": safe(row.get("primary_impact_dimension")),
            "description":      safe(row.get("technical_description")),
            "status":           safe(row.get("approval_status")),
        })
    print(f"  Loading {len(records)} ECO node(s) …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (n:ECO {id: r.id})
        SET n += r
    """, records)


# ══════════════════════════════════════════════════════════════════════════════
# RELATIONSHIP LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_contains_rels(session, graph_edges: pd.DataFrame, bom: pd.DataFrame):
    """
    (Product)-[:CONTAINS]->(Subsystem) and (Subsystem)-[:CONTAINS]->(Component)
    from graph_edges, enriched with bom_relationships data.

    BOM cols: relationship_id, parent_id, parent_type, child_id, child_type,
              relationship_type, quantity, fastener_required
    """
    bom_lu = {}
    for _, row in bom.iterrows():
        key = (safe(row.get("parent_id")), safe(row.get("child_id")))
        qty = safe(row.get("quantity"))
        fast = safe(row.get("fastener_required"))
        if key[0] and key[1]:
            bom_lu[key] = {"quantity": qty, "fastener_required": fast}

    contains_edges = graph_edges[graph_edges["relationship_type"] == "CONTAINS"]

    prod_sub, sub_comp = [], []
    for _, row in contains_edges.iterrows():
        src      = safe(row.get("source_node_id"))
        tgt      = safe(row.get("target_node_id"))
        src_type = safe(row.get("source_node_type"))
        props_raw = row.get("props", {})
        qty  = props_raw.get("quantity") if isinstance(props_raw, dict) else None
        # enrich with bom data if available
        bom_data = bom_lu.get((src, tgt), {})
        fast = bom_data.get("fastener_required")
        if qty is None:
            qty = bom_data.get("quantity")

        rec = {"src": src, "tgt": tgt, "quantity": qty, "fastener_required": fast}
        if src_type == "PRODUCT":
            prod_sub.append(rec)
        elif src_type == "SUBSYSTEM":
            sub_comp.append(rec)

    print(f"  Creating {len(prod_sub)} CONTAINS (Product→Subsystem) …")
    run_batched(session, """
        UNWIND $rows AS r
        MATCH (a:Product {id: r.src}), (b:Subsystem {id: r.tgt})
        MERGE (a)-[rel:CONTAINS]->(b)
        SET rel.quantity = r.quantity, rel.fastener_required = r.fastener_required
    """, prod_sub)

    print(f"  Creating {len(sub_comp)} CONTAINS (Subsystem→Component) …")
    run_batched(session, """
        UNWIND $rows AS r
        MATCH (a:Subsystem {id: r.src}), (b:Component {id: r.tgt})
        MERGE (a)-[rel:CONTAINS]->(b)
        SET rel.quantity = r.quantity, rel.fastener_required = r.fastener_required
    """, sub_comp)


def load_graph_edge_rels(session, graph_edges: pd.DataFrame):
    """
    Load MADE_OF, MANUFACTURED_BY, CAPABLE_OF, GOVERNED_BY, SUPPLIED_BY
    directly from graph_edges.

    FIX: edge properties parsed from properties_json (e.g. cpk, lead_time_days)
    are now stored on the relationship via `SET rel += r.props`.
    """
    # relationship_type → (source_label, target_label)
    rel_map = {
        "MADE_OF":          ("Component", "Material"),
        "MANUFACTURED_BY":  ("Component", "Process"),
        "CAPABLE_OF":       ("Machine",   "Process"),
        "GOVERNED_BY":      ("Process",   "SOP"),
        "SUPPLIED_BY":      ("Material",  "Supplier"),
    }

    for rel_type, (src_label, tgt_label) in rel_map.items():
        subset = graph_edges[graph_edges["relationship_type"] == rel_type]
        records = []
        for _, row in subset.iterrows():
            src = safe(row.get("source_node_id"))
            tgt = safe(row.get("target_node_id"))
            props = clean_props(row.get("props", {}))
            if src and tgt:
                records.append({"src": src, "tgt": tgt, "props": props})

        if not records:
            continue

        # Labels / rel type come from the fixed whitelist above — not user input
        print(f"  Creating {len(records)} {rel_type} ({src_label}→{tgt_label}) …")
        q = f"""
            UNWIND $rows AS r
            MATCH (a:{src_label} {{id: r.src}}), (b:{tgt_label} {{id: r.tgt}})
            MERGE (a)-[rel:{rel_type}]->(b)
            SET rel += r.props
        """
        run_batched(session, q, records)


def load_component_sop_rels(session, components: pd.DataFrame):
    """(Component)-[:GOVERNED_BY]->(SOP) from components.governing_sop_id"""
    records = []
    for _, row in components.iterrows():
        cid = safe(row.get("component_id"))
        sid = safe(row.get("governing_sop_id"))
        if cid and sid:
            records.append({"cid": cid, "sid": sid})
    print(f"  Creating {len(records)} GOVERNED_BY (Component→SOP) …")
    run_batched(session, """
        UNWIND $rows AS r
        MATCH (c:Component {id: r.cid}), (s:SOP {id: r.sid})
        MERGE (c)-[:GOVERNED_BY]->(s)
    """, records)


def load_eco_rels(session, eco: pd.DataFrame):
    """ECO → CHANGED, FROM_MATERIAL, TO_MATERIAL, FROM_MACHINE, TO_MACHINE."""
    changed, from_mat, to_mat, from_mach, to_mach = [], [], [], [], []

    for _, row in eco.iterrows():
        eid = safe(row.get("change_id"))
        tgt = safe(row.get("target_node_id"))
        ct  = safe(row.get("change_type"))
        cur = safe(row.get("current_value"))
        prop = safe(row.get("proposed_value"))

        if eid and tgt:
            changed.append({"eid": eid, "cid": tgt})

        # MATERIAL_SUBSTITUTION: current_value & proposed_value are material IDs
        if ct == "MATERIAL_SUBSTITUTION":
            if eid and cur and str(cur).startswith("MAT-"):
                from_mat.append({"eid": eid, "mid": cur})
            if eid and prop and str(prop).startswith("MAT-"):
                to_mat.append({"eid": eid, "mid": prop})

        # MACHINE_REPLACEMENT: current_value & proposed_value are machine IDs
        elif ct == "MACHINE_REPLACEMENT":
            if eid and cur and str(cur).startswith("MCH-"):
                from_mach.append({"eid": eid, "mid": cur})
            if eid and prop and str(prop).startswith("MCH-"):
                to_mach.append({"eid": eid, "mid": prop})

    print(f"  Creating {len(changed)} CHANGED (ECO→Component) …")
    run_batched(session, """
        UNWIND $rows AS r
        MATCH (e:ECO {id: r.eid}), (c:Component {id: r.cid})
        MERGE (e)-[:CHANGED]->(c)
    """, changed)

    print(f"  Creating {len(from_mat)} FROM_MATERIAL …")
    run_batched(session, """
        UNWIND $rows AS r
        MATCH (e:ECO {id: r.eid}), (m:Material {id: r.mid})
        MERGE (e)-[:FROM_MATERIAL]->(m)
    """, from_mat)

    print(f"  Creating {len(to_mat)} TO_MATERIAL …")
    run_batched(session, """
        UNWIND $rows AS r
        MATCH (e:ECO {id: r.eid}), (m:Material {id: r.mid})
        MERGE (e)-[:TO_MATERIAL]->(m)
    """, to_mat)

    print(f"  Creating {len(from_mach)} FROM_MACHINE …")
    run_batched(session, """
        UNWIND $rows AS r
        MATCH (e:ECO {id: r.eid}), (m:Machine {id: r.mid})
        MERGE (e)-[:FROM_MACHINE]->(m)
    """, from_mach)

    print(f"  Creating {len(to_mach)} TO_MACHINE …")
    run_batched(session, """
        UNWIND $rows AS r
        MATCH (e:ECO {id: r.eid}), (m:Machine {id: r.mid})
        MERGE (e)-[:TO_MACHINE]->(m)
    """, to_mach)


def load_gt_impacts(session, labels: pd.DataFrame):
    """
    GT_IMPACTS edges are the ANSWER KEY for evaluation / GNN training.
    *** Agent traversal queries must NEVER use GT_IMPACTS. ***

    Labels CSV cols:
      impact_label_id, change_id, affected_node_id, affected_node_type, is_affected,
      impact_severity, impact_dimension, confidence_score, label_source, rationale,
      requires_human_review
    """
    affected = labels[labels["is_affected"].astype(str).str.strip() == "1"].copy()
    affected.columns = affected.columns.str.strip()

    # CSV uses uppercase: COMPONENT, PROCESS, MACHINE, SOP
    # Map uppercase CSV value → Neo4j label (from fixed vocabulary, no user input)
    label_map = {
        "COMPONENT": "Component",
        "PROCESS":   "Process",
        "MACHINE":   "Machine",
        "SOP":       "SOP",
    }

    records = []
    for _, row in affected.iterrows():
        eid      = safe(row.get("change_id"))
        nid      = safe(row.get("affected_node_id"))
        nlabel_csv = safe(row.get("affected_node_type"))
        nlabel   = label_map.get(str(nlabel_csv).strip()) if nlabel_csv else None
        sev      = safe(row.get("impact_severity"))
        dim      = safe(row.get("impact_dimension"))
        conf     = safe(row.get("confidence_score"))
        rat      = safe(row.get("rationale"))
        if eid and nid and nlabel:
            records.append({
                "eid": eid, "nid": nid, "nlabel": nlabel,
                "severity": sev, "dimension": dim,
                "confidence": conf, "rationale": rat,
            })

    print(f"  Creating {len(records)} GT_IMPACTS …  (answer key — do NOT traverse in agents)")

    for neo4j_label in label_map.values():
        label_recs = [r for r in records if r["nlabel"] == neo4j_label]
        if not label_recs:
            continue
        # node/relationship names from fixed vocabulary above — no user input
        q = f"""
            UNWIND $rows AS r
            MATCH (e:ECO {{id: r.eid}}), (n:{neo4j_label} {{id: r.nid}})
            MERGE (e)-[rel:GT_IMPACTS]->(n)
            SET rel.severity   = r.severity,
                rel.dimension  = r.dimension,
                rel.confidence = r.confidence,
                rel.rationale  = r.rationale
        """
        run_batched(session, q, label_recs)


# ══════════════════════════════════════════════════════════════════════════════
# STEP-2 SPEC ADDITIONS
#   QualityRequirement, MUST_MEET, SIMILAR_TO, PART_OF, REQUIRES, Component-SUPPLIED_BY
# ══════════════════════════════════════════════════════════════════════════════

def load_quality_requirements(session, components: pd.DataFrame, sops: pd.DataFrame):
    """
    QualityRequirement nodes + (Component)-[:MUST_MEET]->(QualityRequirement).

    There is no quality-requirement CSV, so they are DERIVED from existing data:
      1. One dimensional-tolerance requirement per component
         (from components.nominal_tolerance_mm).
      2. One requirement per SOP (from sops.critical_tolerances), attached to every
         component governed by that SOP.
    """
    # 1. per-component tolerance requirements
    comp_qr = []
    for _, row in components.iterrows():
        cid = safe(row.get("component_id"))
        if not cid:
            continue
        tol = safe(row.get("nominal_tolerance_mm"))
        comp_qr.append({
            "id":   f"QR-TOL-{cid}",
            "cid":  cid,
            "type": "dimensional_tolerance",
            "tolerance_mm": float(tol) if tol is not None else None,
        })

    print(f"  Creating {len(comp_qr)} QualityRequirement (component tolerance) + MUST_MEET …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (q:QualityRequirement {id: r.id})
        SET q.type = r.type,
            q.tolerance_mm = r.tolerance_mm,
            q.source = 'components.nominal_tolerance_mm'
        WITH q, r
        MATCH (c:Component {id: r.cid})
        MERGE (c)-[:MUST_MEET]->(q)
    """, comp_qr)

    # 2. per-SOP critical-tolerance requirements
    sop_qr = []
    for _, row in sops.iterrows():
        sid = safe(row.get("sop_id"))
        if not sid:
            continue
        sop_qr.append({
            "id":   f"QR-SOP-{sid}",
            "sid":  sid,
            "type": "sop_critical_tolerance",
            "spec": safe(row.get("critical_tolerances")),
            "risk": safe(row.get("risk_classification")),
        })

    print(f"  Creating {len(sop_qr)} QualityRequirement (SOP critical tolerance) + MUST_MEET …")
    run_batched(session, """
        UNWIND $rows AS r
        MERGE (q:QualityRequirement {id: r.id})
        SET q.type = r.type,
            q.spec = r.spec,
            q.risk = r.risk,
            q.source = 'sops.critical_tolerances'
        WITH q, r
        MATCH (c:Component)-[:GOVERNED_BY]->(:SOP {id: r.sid})
        MERGE (c)-[:MUST_MEET]->(q)
    """, sop_qr)


def load_eco_similarity(session):
    """
    (ECO)-[:SIMILAR_TO]->(ECO)

    Rule-based, built ONLY from ECO input attributes — never from GT_IMPACTS
    (that would leak the answer key to the agent).

    Two ECOs are similar if they share the same change_type AND either
      - target the same component, or
      - share the same change_family and impact_dimension.
    Stored once per pair (a.id < b.id) with a `basis` property; the agent can
    traverse it in either direction with an undirected pattern.
    """
    print("  Creating SIMILAR_TO (ECO↔ECO, same target) …")
    session.run("""
        MATCH (a:ECO), (b:ECO)
        WHERE a.id < b.id
          AND a.change_type = b.change_type
          AND a.target_node_id = b.target_node_id
        MERGE (a)-[s:SIMILAR_TO]->(b)
        SET s.basis = 'same_change_type_and_target', s.score = 1.0
    """)

    print("  Creating SIMILAR_TO (ECO↔ECO, same family + impact dimension) …")
    session.run("""
        MATCH (a:ECO), (b:ECO)
        WHERE a.id < b.id
          AND a.change_type = b.change_type
          AND a.family = b.family
          AND a.impact_dimension = b.impact_dimension
          AND NOT (a)-[:SIMILAR_TO]->(b)
        MERGE (a)-[s:SIMILAR_TO]->(b)
        SET s.basis = 'same_change_type_family_dimension', s.score = 0.5
    """)


def load_spec_alias_rels(session):
    """
    Add the relationship names/directions exactly as written in the Step-2 spec,
    alongside the existing ones (existing CONTAINS / CAPABLE_OF / Material-SUPPLIED_BY
    are kept, so nothing already built is broken).

      (Component)-[:PART_OF]->(Assembly)   Subsystem nodes also get the :Assembly label
      (Process)-[:REQUIRES]->(Machine)     reverse of Machine-CAPABLE_OF->Process
      (Component)-[:SUPPLIED_BY]->(Supplier)  via Component-MADE_OF->Material-SUPPLIED_BY->Supplier
    """
    print("  Adding :Assembly label to Subsystem nodes …")
    session.run("MATCH (s:Subsystem) SET s:Assembly")

    print("  Creating PART_OF (Component→Assembly) …")
    session.run("""
        MATCH (s:Subsystem)-[:CONTAINS]->(c:Component)
        MERGE (c)-[:PART_OF]->(s)
    """)

    print("  Creating REQUIRES (Process→Machine) …")
    session.run("""
        MATCH (m:Machine)-[r:CAPABLE_OF]->(p:Process)
        MERGE (p)-[req:REQUIRES]->(m)
        SET req.cpk = r.cpk
    """)

    print("  Creating SUPPLIED_BY (Component→Supplier) …")
    session.run("""
        MATCH (c:Component)-[:MADE_OF]->(:Material)-[sb:SUPPLIED_BY]->(s:Supplier)
        MERGE (c)-[cs:SUPPLIED_BY]->(s)
        SET cs.lead_time_days = sb.lead_time_days
    """)


# ══════════════════════════════════════════════════════════════════════════════
# WIPE
# ══════════════════════════════════════════════════════════════════════════════

def wipe_graph(driver):
    print("Wiping graph …")
    with driver.session() as s:
        s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS")
    print("Graph wiped.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build Manufacturing ECO Knowledge Graph")
    parser.add_argument("--wipe", action="store_true",
                        help="Delete all nodes/rels before loading")
    args = parser.parse_args()

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {URI}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to Neo4j: {e}")
        sys.exit(1)

    if args.wipe:
        wipe_graph(driver)

    create_constraints(driver)
    data = load_all_data()

    with driver.session() as s:
        print("\n── Loading nodes ──────────────────────────────────────────")
        load_product_subsystem_nodes(s, data["graph_edges"])
        load_component_nodes(s, data["components"], data["prod_comp"])
        load_material_nodes(s, data["materials"])
        load_process_nodes(s, data["processes"])
        load_machine_nodes(s, data["machines"], data["prod_mach"])
        load_supplier_nodes(s, data["suppliers"])
        load_sop_nodes(s, data["sops"])
        load_eco_nodes(s, data["eco"])

        print("\n── Loading relationships ───────────────────────────────────")
        load_contains_rels(s, data["graph_edges"], data["bom"])
        load_graph_edge_rels(s, data["graph_edges"])
        load_component_sop_rels(s, data["components"])
        load_eco_rels(s, data["eco"])
        load_gt_impacts(s, data["labels"])

        print("\n── Step-2 spec additions ───────────────────────────────────")
        load_quality_requirements(s, data["components"], data["sops"])
        load_eco_similarity(s)
        load_spec_alias_rels(s)

    driver.close()
    print("\n✅  Knowledge graph build complete.")


if __name__ == "__main__":
    main()
