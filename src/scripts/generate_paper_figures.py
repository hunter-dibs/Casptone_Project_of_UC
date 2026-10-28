#!/usr/bin/env python3
"""Export publication figures from saved ADNI results; never fit models.

Run from any directory with the modeling environment:
    .venv-v2/bin/python scripts/generate_paper_figures.py

The output contains aggregate chart data only, with no participant identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "paper-figures-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


BLUE, ORANGE, GREEN, PURPLE = "#0072B2", "#D55E00", "#009E73", "#8B61A8"
GRAY, INK = "#8A949E", "#243746"
COLORS = [BLUE, ORANGE, GREEN, PURPLE, "#CC79A7"]
BASE = "artifactsv2/primary"
APOE = "artifactsv2/apoe_primary"
BIO = "artifactsv2/biomarkers_primary"
SUITE = "artifactsv2/suite_primary"
ASUITE = "artifactsv2/suite_apoe_primary"
ORIGINAL = "reports/modeling/complete_cases"
BASELINE_MODELS = ("logistic_regression", "random_forest", "xgboost")
MODEL_NAMES = {"logistic_regression": "LR", "random_forest": "Random Forest", "xgboost": "XGBoost"}
MODEL_COLORS = {"logistic_regression": BLUE, "random_forest": GREEN, "xgboost": ORANGE}
MODALITIES = {"clinical": "C", "APOE": "A", "MRI": "M", "CSF": "S"}
FEATURES = {
    "FAQTOTAL": "FAQ total", "TOTAL13": "ADAS-13", "MMSCORE": "MMSE",
    "PTGENDER": "Recorded sex/gender", "PTEDUCAT": "Education", "AGE_APPROX": "Approximate age",
    "E4_COUNT": "APOE ε4 count", "LIMMTOTAL": "Logical Memory immediate",
    "LDELTOTAL": "Logical Memory delayed", "AVTOTB": "RAVLT list B",
    "AVDEL30MIN": "RAVLT delayed recall", "AVDELTOT": "RAVLT recognition",
    "ST29SV": "Left hippocampal volume", "ST88SV": "Right hippocampal volume",
    "ST10CV": "Intracranial volume", "ST24CV": "Left entorhinal volume",
    "ST83CV": "Right entorhinal volume", "ST24TA": "Left entorhinal thickness",
    "ST83TA": "Right entorhinal thickness", "ABETA42": "Aβ42", "TAU": "Total tau", "PTAU": "p-tau181",
}
FEATURES.update({f"AVTOT{i}": f"RAVLT trial {i}" for i in range(1, 7)})


def modality(feature):
    if feature.startswith("UCSFFS"):
        return "MRI"
    if feature.startswith("UPENN"):
        return "CSF"
    if feature.startswith("APOE"):
        return "APOE"
    return "Clinical"


MODALITY_COLORS = {"Clinical": BLUE, "MRI": GREEN, "CSF": ORANGE, "APOE": PURPLE}


class FigureWriter:
    def __init__(self, output, pdf_output, dpi):
        self.output, self.pdf_output, self.dpi = output, pdf_output, dpi
        for directory in (output, output / "data", pdf_output):
            directory.mkdir(parents=True, exist_ok=True)
        self.sources = {}
        self.cache = {}
        self.records = []
        self.atlas_path = pdf_output / "paper-figures.pdf"
        self.atlas = PdfPages(self.atlas_path, metadata={"Title": "ADNI methodology and results figures", "Author": "ADNI research project"})

    def read(self, path):
        path = str(path)
        if path not in self.cache:
            full = ROOT / path
            self.sources[path] = hashlib.sha256(full.read_bytes()).hexdigest()
            self.cache[path] = pd.read_csv(full) if full.suffix == ".csv" else json.loads(full.read_text())
        return self.cache[path].copy() if isinstance(self.cache[path], pd.DataFrame) else self.cache[path]

    def data(self, stem, data):
        path = self.output / "data" / f"{stem}.csv"
        pd.DataFrame(data).to_csv(path, index=False)

    def save(self, fig, stem, title, caption, sections, sources, note=""):
        fig.suptitle(title, fontsize=12, fontweight="bold", color=INK)
        if note:
            fig.supxlabel(note, fontsize=8, color=INK)
        # Execute the layout engine before exporting so every format uses the same placement.
        fig.canvas.draw()
        for ext in ("png", "svg"):
            fig.savefig(self.output / f"{stem}.{ext}", dpi=self.dpi, facecolor="white")
        fig.savefig(self.pdf_output / f"{stem}.pdf", facecolor="white", metadata={"Title": title})
        self.atlas.savefig(fig, facecolor="white")
        self.records.append(dict(stem=stem, title=title, caption=caption, sections=sections, sources=sources, note=note))
        plt.close(fig)
        print(f"Created {stem}", flush=True)

    def finish(self):
        self.atlas.close()
        def relative(path):
            return os.path.relpath(path, self.output)
        lines = ["# Paper figures", "", "Generated from the repository's saved results for Sections 04 and 05. No preparation, training, SHAP estimation, software tests, or browser sessions are run by this script.", "", "C = clinical/cognitive; A = APOE; M = FreeSurfer 5.1 MRI; S = CSF. Unless explicitly marked same-day or original, results use the primary retrospective policy. sMCI means supported non-conversion under the documented 36-month endpoint.", "", "Figures 5.1 and 5.2 retain the numbering already used in the results chapter. Figure 4.3 avoids the existing proposed-architecture Figure 4.2. Other numbers are suggested placements. Cohorts overlap; their counts must not be summed. Comparisons across cohorts are descriptive. Bootstrap intervals condition on saved predictions, omit retraining uncertainty, and have no multiplicity correction.", "", f"[All figures as PDF]({relative(self.atlas_path)}) · [Visual gallery](index.html) · [Source manifest](manifest.json)", "", "PNG files use 300 dpi by default; SVG and PDF files are vector exports. Chart values are available under `data/`, without participant identifiers. Source paths and SHA-256 hashes are recorded in the manifest. Use the source CSVs, rather than rounded manuscript values, for numerical reproduction.", ""]
        cards = []
        for r in self.records:
            s = r["stem"]
            links = f"[PNG]({s}.png) · [SVG]({s}.svg) · [PDF]({relative(self.pdf_output / (s + '.pdf'))}) · [Chart data](data/{s}.csv)"
            srcs = ", ".join(f"[{p}]({relative(ROOT / p)})" for p in r["sources"])
            lines += [f"## {r['title']}", "", f"Placement: {r['sections']}", "", f"![{r['title']}]({s}.png)", "", r["caption"], "", links, "", f"Sources: {srcs}", ""]
            cards.append(f'<article><h2>{html.escape(r["title"])}</h2><p>{html.escape(r["sections"])}</p><a href="{s}.svg"><img src="{s}.png" alt="{html.escape(r["title"])}" loading="lazy"></a><p>{html.escape(r["caption"])}</p><p><a href="{s}.png">PNG</a> · <a href="{s}.svg">SVG</a> · <a href="{html.escape(relative(self.pdf_output / (s + ".pdf")))}">PDF</a> · <a href="data/{s}.csv">Chart data</a></p></article>')
        (self.output / "README.md").write_text("\n".join(lines))
        (self.output / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>ADNI paper figures</title><style>body{font:16px/1.6 system-ui;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#243746;background:#f6f8fa}article{background:white;padding:1.5rem;margin:2rem 0;border:1px solid #dce3e9;border-radius:8px}img{width:100%;height:auto}a{color:#0072b2}h1,h2{line-height:1.3}</style><h1>ADNI methodology and results figures</h1><p>Aggregate saved results. Cohorts overlap and differ in membership. See <a href="README.md">the caption and provenance guide</a> for reporting boundaries.</p>' + "".join(cards) + '</html>')
        manifest = {"script": "scripts/generate_paper_figures.py", "methodology": "04-methodology.md", "results": "05-results.md", "dpi": self.dpi, "figure_count": len(self.records), "models_fitted": False, "participant_identifiers_exported": False, "versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__, "scikit-learn": importlib.metadata.version("scikit-learn")}, "source_sha256": self.sources, "figures": self.records}
        (self.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def figure(*args, **kwargs):
    return plt.subplots(*args, layout="constrained", **kwargs)


def style(ax, axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=axis, color="#E5E9ED", lw=0.6)
    ax.set_axisbelow(True)


def select(w, path, experiment, model):
    d = w.read(path)
    d = d[(d.experiment == experiment) & (d.model == model)]
    if len(d) != 1:
        raise ValueError(f"Expected one result for {path}: {experiment}/{model}; got {len(d)}")
    row = d.iloc[0].copy()
    if "accuracy" not in row:
        row["accuracy"] = (row.tp + row.tn) / row.participants
    row["source"] = path
    return row


def predictions(w, path, experiment, model):
    d = w.read(path)
    d = d[(d.experiment == experiment) & (d.model == model)].copy()
    if d.empty or d.RID.duplicated().any() or set(d.label) != {0, 1}:
        raise ValueError(f"Incomplete or duplicate held-out predictions: {path}/{experiment}/{model}")
    return d


def flow_figures(w):
    stem = "figure-4-1-methodological-framework"
    fig, ax = figure(figsize=(9, 7.5))
    ax.set(xlim=(0, 10), ylim=(-0.5, 10)); ax.axis("off")
    nodes = [
        (5, 9.3, 7.8, 0.65, "ADNI source tables and dictionaries", BLUE),
        (5, 8.2, 7.8, 0.7, "Participant identity + explicit baseline MCI", BLUE),
        (2.5, 6.8, 4.1, 1.1, "Predictor preparation\nTiming, quality, modality availability", GREEN),
        (7.5, 6.8, 4.1, 1.1, "Separate diagnostic audit\nDocumented 36-month pMCI / sMCI labels", ORANGE),
        (5, 5.3, 7.8, 0.8, "Complete-case cohorts and matched feature subsets", BLUE),
        (5, 4.05, 7.8, 0.8, "Five participant-level stratified outer folds", BLUE),
        (5, 2.7, 7.8, 1.0, "Outer training participants only\nFit preprocessing + model; three inner folds when selection is required", GREEN),
        (5, 1.35, 7.8, 0.75, "Pool outer held-out probabilities, one per participant and configuration", BLUE),
        (2.5, 0.2, 4.1, 0.7, "Performance, calibration, paired intervals", BLUE),
        (7.5, 0.2, 4.1, 0.7, "Held-out SHAP for selected LR / XGBoost", PURPLE),
    ]
    for x, y, width, height, text, color in nodes:
        ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height, boxstyle="round,pad=0.08", facecolor=color+"12", edgecolor=color, lw=1))
        ax.text(x, y, text, ha="center", va="center", fontsize=9)
    for a, b in [(0,1),(1,2),(1,3),(2,4),(3,4),(4,5),(5,6),(6,7),(7,8),(7,9)]:
        x,y,_,h,*_ = nodes[a]; xx,yy,_,hh,*_ = nodes[b]
        ax.add_patch(FancyArrowPatch((x,y-h/2-0.08),(xx,yy+hh/2+0.08),arrowstyle="-|>",mutation_scale=11,color=GRAY,lw=1))
    data = [{"stage":i+1,"description":n[4].replace("\n", "; ")} for i,n in enumerate(nodes)]
    w.data(stem,data)
    w.save(fig,stem,"Figure 4.1. Methodological framework","Outcome construction and predictor preparation remain separate until complete-case dataset assembly. Learned preprocessing and model selection use training participants. All reported predictions are held out at the participant level. Inner validation applies only where selection or combination is required.","Section 4.1",["04-methodology.md"],"Implemented prediction workflow; the proposed dashboard is outside this diagram.")

    stem = "figure-4-3-temporal-availability"
    fig, axes = figure(3,1,figsize=(8,5.4),sharex=True)
    policies = [("Strict",0,"Processing known, after collection, and by baseline"),("Primary retrospective",-1,"Processing known and after collection; may occur after baseline"),("Same-day sensitivity",0,"Processing known and after collection; may occur after baseline")]
    for ax,(label,end,note) in zip(axes,policies):
        ax.axvspan(0,18,color="#EDF0F2")
        ax.axvline(0,color=INK,lw=1,ls="--")
        ax.plot([-90,end],[0.45,0.45],lw=9,color=BLUE,solid_capstyle="butt")
        ax.scatter([-90,end],[0.45,0.45],color=BLUE,zorder=3,s=30)
        ax.text(-90,0.68,label,fontweight="bold",fontsize=10)
        ax.text(-90,0.1,note,fontsize=8.5)
        ax.text(end-1,0.57,f"Day {end}",ha="right",fontsize=8)
        ax.set(ylim=(-0.05,0.92),yticks=[],xlim=(-96,18))
        ax.spines[["left","right","top"]].set_visible(False)
    axes[-1].set(xticks=[-90,-60,-30,0],xlabel="MRI acquisition / CSF collection relative to diagnostic baseline (days)")
    w.data(stem,[dict(policy=p,start_day=-90,end_day=e,processing_rule=n) for p,e,n in policies])
    w.save(fig,stem,"Figure 4.3. Temporal availability policies","MRI and CSF collection must fall within the displayed inclusive windows. Strict eligibility additionally requires processing by baseline; retrospective eligibility permits later processing. Clinical/APOE predictors and diagnostic outcomes are frozen across policies. A date-only baseline cannot establish within-day ordering.","Section 4.3.4; Table 4.1",["04-methodology.md"],"Baseline = day 0. Later processing does not demonstrate historical result availability.")


def ablation_and_shap(w):
    stem = "figure-5-1-trimodal-ablation"
    exps = ["trimodal_clinical","trimodal_clinical_mri","trimodal_clinical_csf","trimodal_fusion"]
    fig,ax = figure(figsize=(7.5,4.7)); data=[]; x=np.arange(4)
    for i,model in enumerate(BASELINE_MODELS):
        rows=[select(w,f"{BASE}/reports/metrics.csv",e,model) for e in exps]
        values=np.array([r.accuracy for r in rows])*100
        bars=ax.bar(x+(i-1)*0.24,values,0.23,color=MODEL_COLORS[model],label=MODEL_NAMES[model])
        ax.bar_label(bars,fmt="%.1f",fontsize=7,padding=3)
        data.extend([r.to_dict() for r in rows])
    ax.set(xticks=x,xticklabels=["C","C+M","C+S","C+M+S"],ylim=(0,100),ylabel="Held-out accuracy (%)")
    ax.legend(ncol=3,loc="upper center",fontsize=8);style(ax)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.1. Feature ablation on matched participants (N=68)","Baseline accuracy at threshold 0.5 across feature subsets on identical participants and outer folds. Random Forest increases from 72.06% with clinical features to 82.35% with C+M+S. The advanced ResNet result is reported separately and is not represented by these bars.","Section 5.3.5; Table 5.8",[f"{BASE}/reports/metrics.csv"],"C = clinical; M = MRI; S = CSF. Every bar uses the same 68 participants.")

    stem="figure-5-2-trimodal-shap"
    path=f"{BASE}/reports/explanations/trimodal_fusion/xgboost_global_importance.csv"
    d=w.read(path).head(15).iloc[::-1].copy()
    d["label"]=d.feature.map(lambda x:FEATURES.get(x.split("__")[-1],x));d["modality"]=d.feature.map(modality)
    fig,ax=figure(figsize=(7.5,5.8))
    bars=ax.barh(d.label,d.mean_abs_shap*100,color=d.modality.map(MODALITY_COLORS))
    ax.bar_label(bars,fmt="%.2f",fontsize=8,padding=3)
    ax.set(xlabel="Mean absolute SHAP contribution (probability percentage points)",xlim=(0,7.6))
    ax.legend(handles=[Patch(color=MODALITY_COLORS[m],label=m) for m in ["Clinical","MRI","CSF"]],loc="lower right",fontsize=8)
    style(ax,"x");w.data(stem,d)
    w.save(fig,stem,"Figure 5.2. Held-out XGBoost importance for C+M+S (N=68)","Top 15 features ranked by mean absolute held-out permutation SHAP contribution to predicted conversion probability. Aβ42 leads this XGBoost model. Colors identify feature modalities. Magnitudes do not establish direction, per-unit effects, or causality; these explanations do not describe the highest-accuracy ResNet model.","Section 5.8.3",[path],"Contributions explain XGBoost probabilities, not the advanced-model accuracy leader.")


def cohorts(w):
    stem="figure-5-3-cohort-availability"
    summary=w.read("reports/summary.json")
    overlap=w.read("new-data/output-v3/reports/overlap.csv")
    order=["clinical","clinical+APOE","MRI","CSF","clinical+MRI","clinical+CSF","MRI+CSF","clinical+MRI+CSF","APOE+CSF","clinical+APOE+CSF","clinical+APOE+MRI+CSF"]
    d=overlap[(overlap.scenario=="retrospective/fs51") & (overlap.definition=="complete")].set_index("modalities").loc[order].reset_index()
    fig,axes=figure(1,2,figsize=(9.5,5.7),gridspec_kw={"width_ratios":[0.9,1.6]})
    a=axes[0]; values=[summary["converters"],summary["nonconverters"],summary["uncertain_outcomes"]]
    bars=a.bar(["pMCI","sMCI","Unresolved"],values,color=[ORANGE,BLUE,GRAY],width=0.65)
    a.bar_label(bars,labels=[f"{v}\n{v/summary['baseline_MCI_master']:.1%}" for v in values],fontsize=9,padding=4)
    a.set(title="A  Baseline-MCI master (N=1,277)",ylabel="Participants",ylim=(0,700));style(a)
    a.tick_params(axis="x",labelsize=8)
    a=axes[1];y=np.arange(len(d));a.barh(y,d.nonconverters,color=BLUE,label="sMCI");a.barh(y,d.converters,left=d.nonconverters,color=ORANGE,label="pMCI")
    for i,r in d.iterrows():a.text(r.labeled+8,i,f"{r.labeled} ({r.converters}/{r.nonconverters})",va="center",fontsize=8)
    a.set(yticks=y,yticklabels=["+".join(MODALITIES[x] for x in s.split("+")) for s in d.modalities],xlabel="Labeled complete cases",xlim=(0,795),title="B  Primary evaluation availability")
    a.invert_yaxis();style(a,"x");a.legend(loc="lower right",fontsize=8)
    d["master_pMCI"]=summary["converters"];d["master_sMCI"]=summary["nonconverters"];d["master_unresolved"]=summary["uncertain_outcomes"]
    w.data(stem,d)
    w.save(fig,stem,"Figure 5.3. Outcome ascertainment and modality availability","Of 1,277 baseline-MCI participants, 708 have resolved outcomes and 569 remain unresolved. Complete-case evaluation populations vary by required modalities. Bar-end labels give total N and pMCI/sMCI counts. Rows overlap and must not be added; the complete four-modality intersection is empty.","Section 5.1.2; Table 5.1",["reports/summary.json","new-data/output-v3/reports/overlap.csv"],"C = clinical; A = APOE; M = MRI; S = CSF. Cohort rows overlap.")


METHODS=[
    ("Fixed LR","baseline","logistic_regression"),("Random Forest","baseline","random_forest"),("XGBoost","baseline","xgboost"),
    ("Extra Trees","ensembles/voting","extra_trees"),("Equal soft voting","ensembles/voting","equal_voting"),("Weighted soft voting","ensembles/voting","weighted_voting"),("Mixed-model stacking","ensembles/stacking","stacking"),
    ("MLP seed 42","deep_learning/mlp","mlp_seed42"),("MLP probability average","deep_learning/mlp","mlp_seed_mean"),
    ("ResNet seed 42","deep_learning/resnet","resnet_seed42"),("ResNet probability average","deep_learning/resnet","resnet_seed_mean"),
    ("TabPFN seed 42","deep_learning/tabpfn","tabpfn_seed42"),("TabPFN probability average","deep_learning/tabpfn","tabpfn_seed_mean"),
    ("Hybrid seed 42","hybrid_attention","hybrid_seed42"),("Hybrid probability average","hybrid_attention","hybrid_seed_mean"),
    ("Linear-feature LR","lr_stacking","lr_linear_view"),("Spline-feature LR","lr_stacking","lr_spline_view"),("Interaction-feature LR","lr_stacking","lr_interactions_view"),("Approximate-RBF LR","lr_stacking","lr_rbf_view"),("LR equal voting","lr_stacking","lr_equal_voting"),("LR stacking","lr_stacking","lr_stacking"),
]


def comparison_heatmap(w):
    stem="figure-5-4-model-comparison"
    columns=[("mri_clinical","C\nMRI-matched\nN=174"),("mri_fusion","C+M\nN=174"),("csf_fusion","C+S\nN=135"),("trimodal_fusion","C+M+S\nN=68"),("apoe_fusion","C+A\nN=222")]
    data=[];matrix=[];sources=set()
    for label,family,model in METHODS:
        values=[]
        for exp,_ in columns:
            base=APOE if exp.startswith("apoe") else BASE
            suite=ASUITE if exp.startswith("apoe") else SUITE
            path=f"{base}/reports/metrics.csv" if family=="baseline" else f"{suite}/{family}/reports/metrics.csv"
            r=select(w,path,exp,model);sources.add(path);values.append(r.accuracy*100)
            data.append(dict(configuration=label,**r.to_dict()))
        matrix.append(values)
    matrix=np.array(matrix)
    fig,ax=figure(figsize=(8.4,9))
    im=ax.imshow(matrix,cmap="YlGnBu",vmin=65,vmax=90,aspect="auto")
    ax.set(xticks=np.arange(5),xticklabels=[x[1] for x in columns],yticks=np.arange(len(METHODS)),yticklabels=[x[0] for x in METHODS])
    ax.tick_params(length=0);ax.xaxis.tick_top()
    for i in range(len(METHODS)):
        for j in range(5):
            best=np.isclose(matrix[i,j],matrix[:,j].max())
            ax.text(j,i,f"{matrix[i,j]:.2f}"+("*" if best else ""),ha="center",va="center",fontsize=8,color="white" if matrix[i,j]>=82 else INK,fontweight="bold" if best else "normal")
    cb=fig.colorbar(im,ax=ax,shrink=0.5,pad=0.03);cb.set_label("Held-out accuracy (%)")
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.4. Baseline and advanced model accuracy","Primary retrospective accuracies for the configurations in Table 5.9. An asterisk marks the largest displayed accuracy within a column. Seed 42 and probability-average ensembles are retained without selecting secondary seeds. Each column is a specified evaluation; only the first two share the same MRI-matched participants. Cross-column differences do not isolate modality effects.","Section 5.5.7; Table 5.9",sorted(sources),"* Column maximum. Read within columns; populations differ across cohort groups.")


def matched_comparisons(w):
    stem="figure-5-5-matched-modality-comparisons"
    configs=[(APOE,["apoe_clinical","apoe_fusion"],["C","C+A"],"A  Clinical/APOE (N=222)"),(BASE,["mri_clinical","mri_fusion"],["C","C+M"],"B  Clinical/MRI (N=174)"),(BASE,["csf_clinical","csf_fusion"],["C","C+S"],"C  Clinical/CSF (N=135)"),(BIO,["biomarker_mri_csf_mri","biomarker_mri_csf_csf","biomarker_mri_csf_fusion"],["M","S","M+S"],"D  MRI/CSF (N=76)")]
    fig,axes=figure(2,2,figsize=(8,6.3));data=[]
    for ax,(root,exps,labels,title) in zip(axes.flat,configs):
        for i,model in enumerate(["logistic_regression","random_forest","xgboost"]):
            rows=[select(w,f"{root}/reports/metrics.csv",e,model) for e in exps]
            ax.plot(np.arange(len(exps)),[r.accuracy*100 for r in rows],"o-",label=MODEL_NAMES[model],color=MODEL_COLORS[model],lw=1.5,ms=5)
            data.extend([r.to_dict() for r in rows])
        ax.set(xticks=np.arange(len(exps)),xticklabels=labels,ylim=(50,95),ylabel="Accuracy (%)",title=title,xlim=(-0.25,len(exps)-0.75));style(ax)
    axes[0,0].legend(fontsize=7,loc="lower left")
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.5. Matched modality comparisons","Baseline feature comparisons reuse participants and folds within each panel. Adding APOE improves LR accuracy, whereas clinical/MRI and clinical/CSF effects vary by algorithm. MRI/CSF fusion produces a marked XGBoost gain on the shared N=76 cohort. Lines connect feature configurations, not longitudinal visits; uncertainty is shown in Figure 5.11.","Sections 5.3.1-5.3.4; Tables 5.5-5.7",[f"{p}/reports/metrics.csv" for p in [APOE,BASE,BIO]],"Participants and folds match within each panel; panels represent different populations.")


# Panel definitions deliberately pair the same selected feature set with its own cohort.
PANELS=[
    (BASE,"mri_fusion",f"{SUITE}/deep_learning/tabpfn","tabpfn_seed42","TabPFN seed 42","C+M (N=174)","mri_clinical"),
    (BASE,"csf_fusion",f"{SUITE}/deep_learning/mlp","mlp_seed42","MLP seed 42","C+S (N=135)","csf_clinical"),
    (BASE,"trimodal_fusion",f"{SUITE}/deep_learning/resnet","resnet_seed42","ResNet seed 42","C+M+S (N=68)","trimodal_clinical"),
    (APOE,"apoe_fusion",None,None,None,"C+A (N=222)","apoe_clinical"),
]


def discrimination(w):
    for kind,num in [("roc",6),("precision-recall",7)]:
        stem=f"figure-5-{num}-{kind}";fig,axes=figure(2,2,figsize=(8,7));data=[];sources=set()
        for ax,(base,exp,advanced,am,al,title,clinical) in zip(axes.flat,PANELS):
            configs=[(base,exp,m,MODEL_NAMES[m],MODEL_COLORS[m],"-") for m in BASELINE_MODELS]
            if advanced:configs.append((advanced,exp,am,al,PURPLE,"-"))
            configs.append((base,clinical,"logistic_regression","Clinical LR",GRAY,"--"))
            for root,e,m,label,color,ls in configs:
                pp=f"{root}/held_out_predictions.csv";mp=f"{root}/reports/metrics.csv";sources.update([pp,mp])
                d=predictions(w,pp,e,m);r=select(w,mp,e,m)
                if kind=="roc":
                    xx,yy,_=roc_curve(d.label,d.probability);score=r.auroc
                    data.extend(dict(panel=title,model=label,false_positive_rate=float(x),true_positive_rate=float(y)) for x,y in zip(xx,yy))
                else:
                    yy,xx,_=precision_recall_curve(d.label,d.probability);score=r.average_precision
                    data.extend(dict(panel=title,model=label,recall=float(x),precision=float(y)) for x,y in zip(xx,yy))
                if kind=="roc":ax.plot(xx,yy,color=color,ls=ls,lw=1.5,label=f"{label} ({score:.3f})")
                else:ax.step(xx,yy,where="post",color=color,ls=ls,lw=1.3,label=f"{label} ({score:.3f})")
            if kind=="roc":
                ax.plot([0,1],[0,1],ls=":",color=GRAY,lw=1);ax.set(xlabel="False positive rate",ylabel="True positive rate")
            else:
                ax.axhline(d.label.mean(),ls=":",color=GRAY,lw=1,label=f"Prevalence ({d.label.mean():.3f})");ax.set(xlabel="Recall (sensitivity)",ylabel="Precision")
            ax.set(xlim=(0,1),ylim=(0,1.03),title=title);style(ax);ax.legend(loc="lower right" if kind=="roc" else "lower left",fontsize=6.8)
        w.data(stem,data)
        metric="AUROC" if kind=="roc" else "non-interpolated AP"
        w.save(fig,stem,f"Figure 5.{num}. {'ROC' if kind=='roc' else 'Precision-recall'} curves on held-out predictions",f"Curves pool the five outer held-out partitions within each stated population. Legend values report saved {metric}. Solid lines use the panel's fused feature set; the dashed clinical LR comparator uses the same participants. Advanced curves use seed 42, without selecting secondary seeds. {'The diagonal is the chance reference.' if kind=='roc' else 'Dotted horizontal lines show each cohort’s converter prevalence; AP is prevalence-dependent.'} Panels use different evaluation populations.","Section 5.6.1; Table 5.10",sorted(sources),f"Legend: model ({metric}). Cohort-specific held-out predictions; no model refitting.")


def calibration(w):
    stem="figure-5-8-calibration";fig,axes=figure(2,2,figsize=(8,6.7));data=[];sources=set()
    for ax,(base,exp,_,_,_,title,clinical) in zip(axes.flat,PANELS):
        path=f"{base}/reports/calibration.csv";sources.add(path);d=w.read(path)
        for e,m,label,color,ls in [(clinical,"logistic_regression","Clinical LR",GRAY,"--"),(exp,"logistic_regression","Fusion LR",BLUE,"-"),(exp,"random_forest","Fusion RF",GREEN,"-"),(exp,"xgboost","Fusion XGBoost",ORANGE,"-")]:
            a=d[(d.experiment==e)&(d.model==m)].sort_values("mean_probability")
            if a.empty:raise ValueError(f"No calibration data for {path}/{e}/{m}")
            ax.plot(a.mean_probability,a.observed_fraction,marker="o",ms=4,color=color,ls=ls,lw=1.3,label=label)
            data.extend(a.assign(panel=title,configuration=label).to_dict("records"))
        ax.plot([0,1],[0,1],ls=":",color=INK,lw=1)
        ax.set(title=title,xlim=(0,1),ylim=(0,1),xlabel="Mean predicted conversion",ylabel="Observed conversion")
        ax.xaxis.set_major_formatter(PercentFormatter(1));ax.yaxis.set_major_formatter(PercentFormatter(1));style(ax)
        ax.legend(fontsize=7,loc="upper left")
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.8. Probability calibration within matched cohorts","Saved quantile calibration summaries for clinical-only LR and fused baseline models. The diagonal represents perfect agreement. Points summarize held-out probabilities without fitting a recalibration model. Sparse groups, especially in N=68, should not be interpreted as precise prospective calibration estimates.","Section 5.6.4; Table 5.13",sorted(sources),"Saved quantile groups; descriptive calibration without post hoc recalibration.")


SELECTED=[
    (f"{SUITE}/deep_learning/resnet","mri_clinical","resnet_seed_mean","C, MRI-matched\nResNet average; N=174"),
    (f"{SUITE}/deep_learning/tabpfn","mri_fusion","tabpfn_seed42","C+M\nTabPFN seed 42; N=174"),
    (f"{SUITE}/deep_learning/mlp","csf_fusion","mlp_seed42","C+S\nMLP seed 42; N=135"),
    (f"{SUITE}/deep_learning/resnet","trimodal_fusion","resnet_seed42","C+M+S\nResNet seed 42; N=68"),
    (APOE,"apoe_fusion","logistic_regression","C+A\nFixed LR; N=222"),
    (BIO,"biomarker_mri_csf_fusion","xgboost","M+S\nXGBoost; N=76"),
]


def confusion(w):
    stem="figure-5-9-confusion-matrices";fig,axes=figure(2,3,figsize=(9,6.1));data=[];sources=set()
    for ax,(root,e,m,label) in zip(axes.flat,SELECTED):
        p=f"{root}/reports/metrics.csv";sources.add(p);r=select(w,p,e,m);data.append(dict(configuration=label,**r.to_dict()))
        counts=np.array([[r.tn,r.fp],[r.fn,r.tp]],dtype=int);rates=counts/counts.sum(axis=1,keepdims=True)
        ax.imshow(rates,vmin=0,vmax=1,cmap="Blues")
        for i in range(2):
            for j in range(2):ax.text(j,i,f"{counts[i,j]}\n({rates[i,j]:.1%})",ha="center",va="center",fontsize=10,color="white" if rates[i,j]>.6 else INK)
        ax.set(xticks=[0,1],xticklabels=["sMCI","pMCI"],yticks=[0,1],yticklabels=["sMCI","pMCI"],xlabel="Predicted",ylabel="Observed",title=label)
        ax.tick_params(length=0)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.9. Confusion matrices at probability threshold 0.5","Cells give participant counts and percentages within observed outcome class. Colors use a shared 0-100% row scale. The selected high-accuracy configurations retain missed converters: for example, clinical-only ResNet averaging identifies 44/55 converters in the MRI-matched cohort, while clinical/MRI TabPFN identifies 40/55. These are different algorithms and feature sets on that population.","Section 5.6.3; Table 5.12",sorted(sources),"Counts (row percentages). Darker cells indicate a larger share of the observed class.")


def absolute_intervals(w):
    stem="figure-5-10-conditional-intervals"
    configs=[(BASE,"mri_clinical","logistic_regression","C, MRI-matched · LR · N=174"),(BASE,"mri_fusion","xgboost","C+M · XGBoost · N=174"),(APOE,"apoe_fusion","logistic_regression","C+A · LR · N=222"),(BIO,"biomarker_mri","logistic_regression","M · LR · N=184"),(BIO,"biomarker_csf","random_forest","S · Random Forest · N=175"),(BIO,"biomarker_mri_csf_fusion","xgboost","M+S · XGBoost · N=76"),(BASE,"trimodal_fusion","random_forest","C+M+S · Random Forest · N=68")]
    fig,axes=figure(1,2,figsize=(10.4,5),sharey=True);data=[];sources=set()
    for ax,metric,title in zip(axes,["accuracy","auroc"],["Accuracy (%)","AUROC"]):
        for i,(root,e,m,label) in enumerate(configs):
            p=f"{root}/reports/confidence_intervals.csv";sources.add(p)
            d=w.read(p);r=d[(d.experiment==e)&(d.model==m)&(d.metric==metric)].iloc[0]
            scale=100 if metric=="accuracy" else 1;x=r.estimate*scale
            ax.errorbar(x,i,xerr=np.array([[r.estimate-r.lower_95],[r.upper_95-r.estimate]])*scale,fmt="o",color=BLUE,capsize=3,ms=5)
            ax.text(1.02,i,f"{x:.2f} [{r.lower_95*scale:.2f}, {r.upper_95*scale:.2f}]" if scale==100 else f"{x:.3f} [{r.lower_95:.3f}, {r.upper_95:.3f}]",transform=ax.get_yaxis_transform(),fontsize=7.2,va="center")
            data.append(dict(configuration=label,source=p,**r.to_dict()))
        ax.set(yticks=np.arange(len(configs)),yticklabels=[c[3] for c in configs],xlabel=title,xlim=(60,95) if metric=="accuracy" else (.6,1));ax.invert_yaxis();style(ax,"x")
    # Shared axes need a single inversion.
    axes[0].set_ylim(len(configs)-0.5,-0.5)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.10. Saved conditional 95% bootstrap intervals","Selected baseline accuracy and AUROC estimates with saved participant-bootstrap intervals. Intervals condition on existing out-of-fold predictions and omit pipeline retraining and model-selection uncertainty. No absolute interval is supplied for advanced configurations whose reports contain only paired intervals. Rows represent different specified cohorts and do not establish a cross-population ranking.","Section 5.7.2; Table 5.15",sorted(sources),"1,000 participant-bootstrap samples; conditional on saved predictions.")


def paired_intervals(w):
    stem="figure-5-11-paired-accuracy-differences";data=[];sources=set()
    baseline=[(APOE,"apoe_fusion","apoe_clinical","logistic_regression","LR: C+A versus C · N=222"),(BASE,"mri_fusion","mri_clinical","logistic_regression","LR: C+M versus C · N=174"),(BASE,"mri_fusion","mri_clinical","xgboost","XGBoost: C+M versus C · N=174"),(BASE,"csf_fusion","csf_clinical","random_forest","RF: C+S versus C · N=135"),(BASE,"trimodal_fusion","trimodal_clinical","random_forest","RF: C+M+S versus C · N=68"),(BIO,"biomarker_mri_csf_fusion","biomarker_mri_csf_mri","xgboost","XGBoost: M+S versus M · N=76"),(BIO,"biomarker_mri_csf_fusion","biomarker_mri_csf_csf","xgboost","XGBoost: M+S versus S · N=76")]
    for root,c,r,m,label in baseline:
        p=f"{root}/reports/paired_differences.csv";sources.add(p);d=w.read(p)
        a=d[(d.candidate==c)&(d.reference==r)&(d.model==m)&(d.metric=="accuracy")].iloc[0]
        data.append(dict(label=label,comparison_type="Modality addition",source=p,estimate=a.estimate,lower_95=a.lower_95,upper_95=a.upper_95))
    advanced=[(f"{SUITE}/deep_learning/resnet","mri_clinical","resnet_seed_mean","ResNet average versus LR · C · N=174"),(f"{SUITE}/deep_learning/tabpfn","mri_fusion","tabpfn_seed42","TabPFN seed 42 versus LR · C+M · N=174"),("artifactsv2/suite_same_day/deep_learning/tabpfn","csf_fusion","tabpfn_seed_mean","Same-day TabPFN average versus LR · C+S · N=200"),(f"{ASUITE}/deep_learning/tabpfn","apoe_csf_clinical_csf","tabpfn_seed42","TabPFN seed 42 versus LR · C+S · N=29")]
    for root,e,m,label in advanced:
        p=f"{root}/reports/paired_vs_logistic.csv";sources.add(p);d=w.read(p)
        a=d[(d.experiment==e)&(d.model==m)&(d.metric=="accuracy")].iloc[0]
        data.append(dict(label=label,comparison_type="Algorithm",source=p,estimate=a.difference,lower_95=a.lower_95,upper_95=a.upper_95))
    fig,ax=figure(figsize=(10.3,6.7))
    for i,r in enumerate(data):
        color=BLUE if r["comparison_type"]=="Modality addition" else PURPLE
        ax.errorbar(r["estimate"]*100,i,xerr=np.array([[r["estimate"]-r["lower_95"]],[r["upper_95"]-r["estimate"]]])*100,fmt="o",color=color,capsize=3,ms=5)
        ax.text(1.02,i,f'{r["estimate"]*100:+.2f} [{r["lower_95"]*100:+.2f}, {r["upper_95"]*100:+.2f}]',transform=ax.get_yaxis_transform(),fontsize=8,va="center")
    ax.axvline(0,color=INK,lw=1,ls="--");ax.axhline(6.5,color=GRAY,lw=.7)
    ax.set(yticks=np.arange(len(data)),yticklabels=[r["label"] for r in data],xlim=(-9,37),xlabel="Paired accuracy difference (percentage points)")
    ax.invert_yaxis();style(ax,"x")
    ax.legend(handles=[Line2D([],[],marker="o",color=BLUE,label="Modality addition"),Line2D([],[],marker="o",color=PURPLE,label="Algorithm comparison")],loc="upper right",fontsize=8)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.11. Paired accuracy differences and conditional intervals","Saved paired differences align predictions for the same participants. Positive values favor the first configuration. Blue rows compare added modalities with the same algorithm; purple rows compare algorithms with fixed features. The N=29 C+S algorithm contrast is restricted to the clinical/APOE/CSF-matched population. Conditional 95% intervals use 1,000 bootstrap samples, omit retraining uncertainty, and are not adjusted for multiple comparisons.","Section 5.7.3; Table 5.16",sorted(sources),"Positive values favor the first configuration; intervals are exploratory and conditional.")


def fold_variability(w):
    stem="figure-5-12-fold-variability"
    configs=[(ORIGINAL,"clinical_reference","logistic_regression","Original C · LR · N=589"),(ORIGINAL,"matched_fusion","logistic_regression","Original C+A · LR · N=222")]+[(f"{r}/reports",e,m,label.replace("\n"," · ")) for r,e,m,label in SELECTED[:4]]
    fig,axes=figure(1,2,figsize=(9.6,5),sharey=True);data=[];sources=set()
    for i,(root,e,m,label) in enumerate(configs):
        path=f"{root}/fold_metrics.csv";mp=f"{root}/metrics.csv";sources.update([path,mp]);d=w.read(path);d=d[(d.experiment==e)&(d.model==m)].copy()
        if "accuracy" not in d:d["accuracy"]=(d.tp+d.tn)/d.participants
        if len(d)!=5:raise ValueError(f"Expected five folds: {path}/{e}/{m}")
        pooled=select(w,mp,e,m)
        data.extend(d.assign(configuration=label,pooled_accuracy=pooled.accuracy,pooled_auroc=pooled.auroc).to_dict("records"))
        for ax,metric,scale in [(axes[0],"accuracy",100),(axes[1],"auroc",1)]:
            vals=d[metric]*scale
            ax.plot([vals.min(),vals.max()],[i,i],color=GRAY,lw=1)
            ax.scatter(vals,i+np.linspace(-.13,.13,5),s=25,c=COLORS,edgecolor="white",lw=.3,zorder=3)
            ax.scatter([pooled[metric]*scale],[i],marker="D",s=37,facecolor="none",edgecolor=INK,zorder=4)
            ax.set(yticks=np.arange(len(configs)),yticklabels=[c[3] for c in configs],xlabel="Fold accuracy (%)" if scale==100 else "Fold AUROC",xlim=(65,102) if scale==100 else (.65,1.02));style(ax,"x")
    axes[0].set_ylim(len(configs)-.5,-.5)
    axes[1].legend(handles=[Line2D([],[],marker="o",ls="",color=COLORS[i],label=f"Fold {i+1}") for i in range(5)]+[Line2D([],[],marker="D",ls="",markerfacecolor="none",color=INK,label="Pooled")],loc="upper left",ncol=3,fontsize=7)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.12. Fold variability versus pooled performance","Colored points display all five held-out fold estimates; open diamonds display pooled estimates from all held-out participants. Horizontal spans show observed fold ranges, not confidence intervals. Shared training participants make folds dependent; a perfect fold does not imply perfect pooled performance.","Section 5.7.1; Table 5.14",sorted(sources),"Colored points = folds; open diamond = pooled estimate. Ranges are descriptive.")


def seed_variability(w):
    stem="figure-5-13-seed-stability"
    configs=[("mlp","mri_clinical","MLP · C, MRI-matched · N=174"),("resnet","mri_clinical","ResNet · C, MRI-matched · N=174"),("tabpfn","mri_fusion","TabPFN · C+M · N=174"),("mlp","csf_fusion","MLP · C+S · N=135"),("mlp","trimodal_fusion","MLP · C+M+S · N=68"),("resnet","trimodal_fusion","ResNet · C+M+S · N=68")]
    fig,ax=figure(figsize=(8.2,4.8));data=[];sources=set()
    for i,(family,e,label) in enumerate(configs):
        p=f"{SUITE}/deep_learning/{family}/reports/metrics.csv";sources.add(p);rows=[select(w,p,e,f"{family}_seed{s}") for s in [42,43,44,"_mean"]]
        vals=np.array([r.accuracy*100 for r in rows]);ax.plot([vals[:3].min(),vals[:3].max()],[i,i],color=GRAY,lw=1)
        for j,r in enumerate(rows):
            ax.scatter(r.accuracy*100,i+(j-1.5)*.11,color=[BLUE,GREEN,ORANGE,PURPLE][j],marker="D" if j==3 else "o",s=40,zorder=3)
            data.append(dict(configuration=label,seed=["42","43","44","probability_average"][j],**r.to_dict()))
    ax.set(yticks=np.arange(len(configs)),yticklabels=[c[2] for c in configs],xlim=(70,91),xlabel="Held-out accuracy (%)");ax.invert_yaxis();style(ax,"x")
    ax.legend(handles=[Line2D([],[],marker="D" if j==3 else "o",ls="",color=[BLUE,GREEN,ORANGE,PURPLE][j],label=["Seed 42","Seed 43","Seed 44","Probability average"][j]) for j in range(4)],loc="upper left",ncol=2,fontsize=8)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.13. Seed and probability-average stability","The three configured seeds and their probability-average predictions use the same participants and outer folds. The ensemble averages probabilities before classification; it does not average accuracies. Averaging improves clinical-only ResNet accuracy in the MRI-matched population, while three-modality ResNet seed 42 exceeds its ensemble. Seed variation is descriptive and does not supply independent validation.","Section 5.7.4; Table 5.17",sorted(sources),"Ensembles average probabilities before thresholding; seeds are not independent cohorts.")


def explanation_panels(w):
    stem="figure-5-14-shap-comparison"
    roots=[(f"{ORIGINAL}/explanations","Original C+A · N=222")]+[(f"{BASE}/reports/explanations/{e}",label) for e,label in [("mri_fusion","Primary C+M · N=174"),("csf_fusion","Primary C+S · N=135"),("trimodal_fusion","Primary C+M+S · N=68")]]
    fig,axes=figure(4,2,figsize=(10.5,12));data=[];sources=set()
    for i,(root,label) in enumerate(roots):
        for j,model in enumerate(["logistic_regression","xgboost"]):
            path=f"{root}/{model}_global_importance.csv";sources.add(path);d=w.read(path).head(8).iloc[::-1].copy();d["label"]=d.feature.map(lambda x: FEATURES.get(x.split("__")[-1],x));d["modality"]=d.feature.map(modality)
            ax=axes[i,j];ax.barh(d.label,d.mean_abs_shap*100,color=d.modality.map(MODALITY_COLORS));ax.set(title=f"{label} · {MODEL_NAMES[model]}",xlim=(0,16),xlabel="Mean |SHAP| (probability pp)");ax.tick_params(axis="y",labelsize=8);style(ax,"x")
            data.extend(d.assign(experiment=label,model=model,source=path).to_dict("records"))
    axes[0,0].legend(handles=[Patch(color=c,label=m) for m,c in MODALITY_COLORS.items()],loc="lower right",fontsize=7)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.14. Model-specific global SHAP importance","Top eight features from completed held-out LR and XGBoost explanations. A shared axis measures mean absolute contribution in conversion-probability percentage points. FAQ and cognitive scores recur, while entorhinal volume leads three-modality LR and Aβ42 leads three-modality XGBoost. These model- and population-specific attributions do not measure causal effects or explain the advanced accuracy leaders.","Section 5.8.1; Table 5.18",sorted(sources),"C = clinical; A = APOE; M = MRI; S = CSF. Only completed LR/XGBoost explanations.")

    stem="figure-5-15-shap-ranking-stability";fig,axes=figure(1,2,figsize=(9.3,5),sharey=True);data=[];labels=[]
    for i,(root,label) in enumerate(roots):
        path=f"{root}/fold_stability.csv";sources.add(path);d=w.read(path)
        for j,model in enumerate(["logistic_regression","xgboost"]):
            y=2*i+j;color=BLUE if j==0 else ORANGE;labels.append(f"{label} · {MODEL_NAMES[model]}")
            a=d[d.model==model]
            data.extend(a.assign(experiment=label,source=path).to_dict("records"))
            for ax,col in zip(axes,["rank_correlation","top_feature_jaccard"]):
                vals=a[col].to_numpy();ax.plot([vals.min(),vals.max()],[y,y],color=color,lw=1)
                ax.scatter(vals,y+np.linspace(-.12,.12,len(vals)),s=17,color=color,alpha=.6);ax.scatter([vals.mean()],[y],marker="D",s=35,facecolor="white",edgecolor=INK,zorder=4)
    for ax,title in zip(axes,["Spearman rank correlation","Top-five Jaccard overlap"]):
        ax.set(yticks=np.arange(8),yticklabels=labels,xlabel=title,xlim=(-.1,1.05));style(ax,"x");ax.axvline(0,color=GRAY,lw=.7)
    axes[0].set_ylim(7.5,-.5)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.15. SHAP ranking consistency across fold pairs","Each row shows the ten pairwise comparisons among five fold-specific explanations. Points are saved rank correlations or top-five Jaccard overlaps; open diamonds are arithmetic means. The smallest three-modality cohort has especially variable LR rankings. Fold models share training participants, and the limited SHAP permutation/background budget contributes to observed variation.","Section 5.8.4; Table 5.19",[f"{r}/fold_stability.csv" for r,_ in roots],"Points = ten fold-pair comparisons; diamond = mean. Descriptive consistency, not replication.")


def sensitivity(w):
    stem="figure-5-16-availability-sensitivity"
    overlap_path="new-data/output-v3/reports/overlap.csv";d=w.read(overlap_path)
    configs=[("biomarker_csf",BIO,"S"),("csf_fusion",BASE,"C+S"),("biomarker_mri_csf_fusion",BIO,"M+S"),("trimodal_fusion",BASE,"C+M+S"),("biomarker_apoe_csf_fusion",BIO,"A+S"),("apoe_csf_fusion",APOE,"C+A+S")]
    summary=w.read("artifactsv2/remaining_models_summary.csv");data=[]
    # Match the chapter's selection: baseline algorithms, seed 42, or the predefined ensemble.
    eligible=summary.model.isin({method[2] for method in METHODS} | {"tuned_logistic_regression"})
    primary=[];same=[];counts=[]
    for e,root,label in configs:
        best=[]
        for policy in ["retrospective","retrospective_same_day"]:
            a=summary[(summary.experiment==e)&(summary.policy==policy)&eligible].sort_values(["accuracy","auroc","brier_score"],ascending=[False,False,True],kind="stable")
            if a.empty:raise ValueError(f"Missing sensitivity results: {e}/{policy}")
            r=a.iloc[0];best.append(r);data.append(dict(kind="selected_performance",feature_set=label,**r.to_dict()))
        primary.append(best[0].accuracy*100);same.append(best[1].accuracy*100);counts.append([best[0].participants,best[1].participants])
    fig,axes=figure(1,3,figsize=(11,5.4),gridspec_kw={"width_ratios":[1,1.15,1.35]})
    y=np.arange(len(configs));a=axes[0]
    for j,(label,color) in enumerate([("Primary",BLUE),("Same-day",ORANGE)]):a.barh(y+(j-.5)*.33,np.array(counts)[:,j],height=.31,color=color,label=label)
    a.set(yticks=y,yticklabels=[c[2] for c in configs],xlabel="Labeled participants",title="A  Population size",xlim=(0,290));a.invert_yaxis();style(a,"x");a.legend(fontsize=7,loc="lower right")
    for i,(p,s) in enumerate(zip(primary,same)):
        axes[1].plot([p,s],[i,i],color=GRAY,lw=1);axes[1].scatter([p,s],[i-.06,i+.06],color=[BLUE,ORANGE],s=28)
    axes[1].set(yticks=y,yticklabels=[c[2] for c in configs],xlabel="Selected accuracy (%)",xlim=(45,93),title="B  Policy-specific results");axes[1].invert_yaxis();style(axes[1],"x")
    mods=["MRI","clinical+MRI","clinical+MRI+CSF","clinical+APOE+MRI+CSF"]
    scenarios=["strict/fs7","strict/fs51","retrospective/fs7","retrospective/fs51","retrospective_same_day/fs7","retrospective_same_day/fs51"]
    a=d[(d.definition=="complete")&d.modalities.isin(mods)].pivot(index="scenario",columns="modalities",values="labeled").loc[scenarios,mods]
    im=axes[2].imshow(a.to_numpy(),cmap="Blues",aspect="auto",vmin=0,vmax=200)
    axes[2].set(xticks=np.arange(4),xticklabels=["M","C+M","C+M+S","C+A+M+S"],yticks=np.arange(6),yticklabels=["Strict · FS7","Strict · FS5.1","Primary · FS7","Primary · FS5.1","Same-day · FS7","Same-day · FS5.1"],title="C  Complete labeled availability")
    axes[2].tick_params(axis="x",rotation=55,labelsize=7);axes[2].tick_params(axis="y",labelsize=7)
    for i in range(6):
        for j in range(4):axes[2].text(j,i,str(a.iloc[i,j]),ha="center",va="center",fontsize=8,color="white" if a.iloc[i,j]>120 else INK)
    data.extend(dict(kind="availability",scenario=s,feature_set=m,labeled=int(a.loc[s,m])) for s in scenarios for m in mods)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.16. Timing and MRI-product availability sensitivity","Panels A and B compare primary and same-day complete-case populations and selected predictive-model performance. Selection uses accuracy, then AUROC, then lower Brier, with seed 42 and probability-average ensembles eligible. Model identities and exact values are recorded in the chart data. Changed membership and sometimes changed algorithms prevent a causal timing interpretation. Panel C reports availability under product-specific rules, not predictive superiority of FreeSurfer versions. The complete four-modality intersection remains empty.","Sections 5.9.1-5.9.2; Tables 5.20-5.21",[overlap_path,"artifactsv2/remaining_models_summary.csv"],"Primary and same-day populations overlap; selected algorithms may differ. FS = FreeSurfer.")


def single_modalities(w):
    stem="figure-5-17-single-modality-metrics"
    configs=[(f"{BIO}/reports","biomarker_mri","MRI · N=184"),(f"{BIO}/reports","biomarker_csf","CSF · N=175"),(ORIGINAL,"matched_apoe","APOE · matched N=222")]
    fig,axes=figure(2,2,figsize=(8.4,6.7));data=[];sources=set()
    models=BASELINE_MODELS
    for ax,metric,title,scale,lim in zip(axes.flat,["accuracy","sensitivity","auroc","brier_score"],["A  Accuracy (%)","B  Sensitivity (%)","C  AUROC","D  Brier score (lower is better)"],[100,100,1,1],[(0,100),(0,100),(0,1),(0,.3)]):
        for j,m in enumerate(models):
            rows=[]
            for root,e,label in configs:
                p=f"{root}/metrics.csv";sources.add(p);r=select(w,p,e,m);rows.append(r)
                if metric=="accuracy":data.append(dict(population=label,**r.to_dict()))
            ax.bar(np.arange(3)+(j-1)*.24,[r[metric]*scale for r in rows],width=.23,color=MODEL_COLORS[m],label=MODEL_NAMES[m])
        ax.set(xticks=np.arange(3),xticklabels=[c[2] for c in configs],ylim=lim,title=title);ax.tick_params(axis="x",labelsize=7);style(ax)
    axes[0,1].legend(fontsize=7,ncol=3)
    w.data(stem,data)
    w.save(fig,stem,"Figure 5.17. Complementary single-modality performance measures","Primary MRI and CSF baseline results and the original matched APOE-only comparison. Accuracy, sensitivity, AUROC, and Brier score reveal different aspects of performance. In particular, MRI LR accuracy of 70.65% coexists with sensitivity of 33.90%. Each modality has its own stated cohort; bars across modalities are not a matched feature comparison.","Section 5.2; Table 5.4",sorted(sources),"Different modality populations; compare algorithms within each modality. Threshold = 0.5.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"figures")
    parser.add_argument("--pdf-dir",type=Path,help="PDF directory (default: OUTPUT_DIR/pdf)")
    parser.add_argument("--dpi",type=int,default=300)
    args=parser.parse_args()
    if args.dpi<72:parser.error("--dpi must be at least 72")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":9.5,"axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"axes.labelcolor":INK,"text.color":INK,"axes.edgecolor":"#778592","pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white","figure.constrained_layout.h_pad":0.13,"figure.constrained_layout.w_pad":0.13})
    w=FigureWriter(args.output_dir.resolve(),(args.pdf_dir or args.output_dir/"pdf").resolve(),args.dpi)
    # Include the exact manuscript versions used when tracing the output's provenance.
    for path in ["04-methodology.md","05-results.md","03-dataset.md","scripts/generate_paper_figures.py"]:w.sources[path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    try:
        flow_figures(w)
        ablation_and_shap(w)
        cohorts(w)
        comparison_heatmap(w)
        matched_comparisons(w)
        discrimination(w)
        calibration(w)
        confusion(w)
        absolute_intervals(w)
        paired_intervals(w)
        fold_variability(w)
        seed_variability(w)
        explanation_panels(w)
        sensitivity(w)
        single_modalities(w)
        w.finish()
    finally:
        w.atlas.close()
        plt.close("all")
    print(f"Exported {len(w.records)} figures to {w.output}; PDF collection: {w.atlas_path}")


if __name__=="__main__":
    main()
