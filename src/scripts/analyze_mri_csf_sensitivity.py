#!/usr/bin/env python3
"""Read-only primary inputs; separate MRI/CSF availability sensitivity, no training."""
import argparse
from collections import Counter
import itertools
import json
from dataset_paths import comparable_config
from pathlib import Path
from prepare_adni import (ROOT, MRI, CSF, BL_CODES, read_csv, write_csv, digest,
                          local_path, group, dt, mri_qc, cleaned, select_measurement)


def present(x):
    return x is not None and x != ''


def main(config='config/mri_csf_sensitivity.json'):
    settings=json.loads(local_path(config).read_text())
    cfg=json.loads(local_path(settings['primary_config']).read_text())
    primary=local_path(cfg['output_dir']); primary_reports=local_path(cfg['report_dir'])
    out=local_path(settings['output_dir']); report=local_path(settings['report_dir'])
    if out==primary or report==primary_reports:
        raise ValueError('Sensitivity outputs must be separate from primary outputs')
    if settings['lookback_days']!=90 or settings['horizon_months']!=36 or settings['exploratory_max_collection_offset_days']!=-1 or settings['require_known_processing_date'] is not True:
        raise ValueError('This documented analysis requires a 90-day window, 36-month labels, strictly earlier collection, and known processing dates')
    manifest=json.loads((primary_reports/'run_manifest.json').read_text())
    if comparable_config(cfg)!=comparable_config(manifest['config']) or cfg['horizon_months']!=36 or cfg['availability_policy']!='result_by_baseline':
        raise ValueError('Primary configuration does not match the saved strict 36-month run')
    if any(cfg['lookback_days'][t]!=90 for t in [MRI,CSF]):
        raise ValueError('Primary MRI/CSF lookbacks must be 90 days')
    source_paths={r['table']:local_path(r['filename']) for r in manifest['inputs']}
    for item in manifest['inputs']:
        if digest(local_path(item['filename']))!=item['sha256']:
            raise ValueError('Raw input changed since primary preparation: '+item['filename'])
    # Snapshot every existing primary file, excluding this separate sensitivity subtree.
    protected=set(p for p in primary.rglob('*') if p.is_file())
    protected.update(p for p in primary_reports.rglob('*') if p.is_file() and not p.is_relative_to(primary_reports/'sensitivity'))
    protected.update(source_paths.values())
    protected.add(local_path(settings['primary_config']))
    protected.add(ROOT/'src/scripts/prepare_adni.py')
    protected.update(ROOT.glob('*.csv'))
    before={str(p.relative_to(ROOT)):digest(p) for p in sorted(protected)}
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    header,primary_rows=read_csv(primary/'baseline_predictors.csv')
    _,outcomes=read_csv(primary/'outcome_audit.csv')
    byout={r['RID']:r for r in outcomes}
    if len(byout)!=len(outcomes) or len({r['RID'] for r in primary_rows})!=len(primary_rows) or set(byout)!={r['RID'] for r in primary_rows}:
        raise ValueError('Primary participant cardinality mismatch')
    schema=json.loads((primary/'predictor_schema.json').read_text())
    modalities=schema['modalities']
    data={}
    for t in [MRI,CSF,'Key_MRI']:
        _,rows=read_csv(source_paths[t])
        for i,r in enumerate(rows,2):r['_row']=i;r['_file']=str(source_paths[t].relative_to(ROOT))
        data[t]=rows
    indexes={t:group(data[t]) for t in [MRI,CSF]}
    image_index=group(data['Key_MRI'],'image_id')
    labeled={r['RID'] for r in outcomes if r['label_provisional'] in {'0','1'}}
    policies={'strict':0,'retrospective_processing':-1}
    datasets={p:[dict(r) for r in primary_rows] for p in policies}
    dsindexes={p:{r['RID']:r for r in rr} for p,rr in datasets.items()}
    audit=[];selection=[];waterfall=[];blockers=[];unknown=[];link_summary=[];same_day=[]
    for t in [MRI,CSF]:
        all_records=[]
        for rid,o in byout.items():
            bd=dt(o['baseline_date'])
            for r in indexes[t].get(rid,[]):
                collection=dt(r['EXAMDATE']); processing=dt(r['RUNDATE'])
                offset=(collection-bd).days if collection else None
                matches=image_index.get(r.get('IMAGEUID','').removeprefix('I'),[]) if t==MRI else []
                link='not_applicable'
                if t==MRI:
                    link='missing_image_uid_match' if not matches else 'multiple_image_uid_matches' if len(matches)>1 else 'matched' if matches[0]['subject_id']==r['PTID'] and matches[0]['image_date']==r['EXAMDATE'] else 'participant_or_date_conflict'
                values,flags=cleaned(t,r,cfg)
                facts={
                    'identifier_agrees':r['PTID']==o['PTID'],
                    'phase_and_visit':r['PHASE']==o['baseline_phase'] and r['VISCODE2'] in BL_CODES,
                    'known_collection_date':collection is not None,
                    'collection_within_90_days':offset is not None and -90<=offset<=0,
                    'collection_strictly_before_baseline':offset is not None and -90<=offset<0,
                    'known_processing_date':processing is not None,
                    'processing_not_before_collection':collection is not None and processing is not None and processing>=collection,
                    'processing_by_baseline':processing is not None and processing<=bd,
                    'quality_control':mri_qc(r) if t==MRI else r.get('HAS_QC_ERROR') in (None,'','0'),
                    'verified_assay_batch':t!=CSF or r['BATCH'] in cfg['csf_verified_batches'],
                    'usable_values':any(v is not None for v in values.values())}
                a={'RID':rid,'modality':'MRI' if t==MRI else 'CSF','source_file':r['_file'],'source_row':r['_row'],
                   'baseline_date':o['baseline_date'],'collection_date':r['EXAMDATE'],'processing_date':r['RUNDATE'],
                   'collection_offset_days':offset if offset is not None else '',
                   'processing_offset_days':(processing-bd).days if processing else '',
                   'processing_lag_days':(processing-collection).days if processing and collection else '',
                   'resolved_outcome':int(rid in labeled),'outcome_reason':o['outcome_reason'],
                   'PHASE':r['PHASE'],'VISCODE2':r['VISCODE2'],'BATCH':r.get('BATCH',''),
                   'STATUS':r.get('STATUS',''),'OVERALLQC':r.get('OVERALLQC',''),'HIPPOQC':r.get('HIPPOQC',''),
                   'IMAGEUID':r.get('IMAGEUID',''),'key_mri_link_status':link,'value_flags':';'.join(flags),
                   **{k:int(v) for k,v in facts.items()}}
                audit.append(a);all_records.append((r,a))
        for policy,cutoff in policies.items():
            # Counts at successive gates are order-dependent; not independent causal effects.
            stages=[('source_records',lambda a:True),
                    ('consistent_identifier',lambda a:a['identifier_agrees']),
                    ('same_phase_initial_visit',lambda a:a['phase_and_visit']),
                    ('known_collection_date',lambda a:a['known_collection_date']),
                    ('collection_window',lambda a:a['collection_within_90_days'] if cutoff==0 else a['collection_strictly_before_baseline']),
                    ('known_processing_date',lambda a:a['known_processing_date']),
                    ('processing_on_or_after_collection',lambda a:a['processing_not_before_collection'])]
            if policy=='strict':stages.append(('processed_by_baseline',lambda a:a['processing_by_baseline']))
            stages.extend([('QC_requirement',lambda a:a['quality_control']),('verified_assay_batch',lambda a:a['verified_assay_batch']),('usable_measurements',lambda a:a['usable_values'])])
            pool=all_records
            previous=set(byout)
            for order,(stage,predicate) in enumerate(stages):
                pool=[(r,a) for r,a in pool if predicate(a)]
                participants={a['RID'] for r,a in pool}
                waterfall.append({'policy':policy,'modality':'MRI' if t==MRI else 'CSF','stage_order':order,'stage':stage,'rows':len(pool),
                                  'participants':len(participants),'resolved_outcome_participants':len(participants&labeled),
                                  'participants_lost_from_previous':len(previous-participants)})
                previous=participants
            # Independently counted reasons among phase/visit candidates. Reasons overlap.
            candidate=[a for r,a in all_records if a['phase_and_visit']]
            reason_pred={
                'missing_or_invalid_collection_date':lambda a:not a['known_collection_date'],
                'collection_after_baseline':lambda a:a['collection_offset_days']!='' and a['collection_offset_days']>0,
                'collection_older_than_90_days':lambda a:a['collection_offset_days']!='' and a['collection_offset_days']<-90,
                'baseline_day_not_proven_before_prediction':lambda a:cutoff<0 and a['collection_offset_days']==0,
                'missing_or_invalid_processing_date':lambda a:not a['known_processing_date'],
                'processing_before_collection':lambda a:a['known_collection_date'] and a['known_processing_date'] and not a['processing_not_before_collection'],
                'processing_after_baseline':lambda a:policy=='strict' and a['processing_offset_days']!='' and a['processing_offset_days']>0,
                'QC_requirement_not_met':lambda a:not a['quality_control'],
                'CSF_batch_methods_unverified':lambda a:not a['verified_assay_batch'],
                'missing_key_MRI_link_audit_only':lambda a:a['key_mri_link_status']=='missing_image_uid_match',
                'unresolved_36_month_outcome':lambda a:not a['resolved_outcome']}
            for reason,predicate in reason_pred.items():
                rr=[a for a in candidate if predicate(a)];ids={a['RID'] for a in rr}
                blockers.append({'policy':policy,'modality':'MRI' if t==MRI else 'CSF','reason':reason,'rows':len(rr),'participants':len(ids),
                                 'participants_with_resolved_outcome':len(ids&labeled),'denominator':'same-phase sc/scmri/bl source candidates; overlapping reasons'})
            selected_ids=set()
            for rid,o in byout.items():
                b={'EXAMDATE':o['baseline_date'],'PHASE':o['baseline_phase']}
                rr=indexes[t].get(rid,[])
                if policy!='strict':
                    # Missing run dates are not treated as evidence of retrospective processing.
                    rr=[r for r in rr if dt(r['RUNDATE']) and dt(r['EXAMDATE']) and dt(r['EXAMDATE'])<dt(o['baseline_date'])]
                v,r,reason,n=select_measurement(t,rr,b,cfg,90,'result_by_baseline' if policy=='strict' else 'acquisition_by_baseline')
                target=dsindexes[policy][rid]
                for f in modalities['MRI' if t==MRI else 'CSF']:
                    value=v.get(f.split('__',1)[1])
                    if policy=='strict':
                        old=target[f]
                        if (present(old)!=present(value)) or (present(old) and float(old)!=float(value)):
                            raise ValueError('Strict reconstruction differs from primary predictor: '+f)
                    target[f]=value
                has=any(present(target[f]) for f in modalities['MRI' if t==MRI else 'CSF'])
                if has:selected_ids.add(rid)
                record=next((a for rr0,a in all_records if r is rr0),{}) if r else {}
                selection.append({'policy':policy,'modality':'MRI' if t==MRI else 'CSF','RID':rid,'source_row':r['_row'] if r else '',
                                  'source_file':r['_file'] if r else '', 'collection_date':r['EXAMDATE'] if r else '',
                                  'processing_date':r['RUNDATE'] if r else '', 'baseline_date':o['baseline_date'],
                                  'collection_offset_days':record.get('collection_offset_days',''),
                                  'processing_offset_days':record.get('processing_offset_days',''),
                                  'processing_lag_days':record.get('processing_lag_days',''),
                                  'key_mri_link_status':record.get('key_mri_link_status',''),
                                  'BATCH':r.get('BATCH','') if r else '', 'selection_reason':reason,'candidate_count':n,
                                  'has_usable_values':int(has),'label_provisional':o['label_provisional'],'outcome_reason':o['outcome_reason']})
            waterfall.append({'policy':policy,'modality':'MRI' if t==MRI else 'CSF','stage_order':len(stages),'stage':'selected_record_with_usable_values',
                              'rows':len(selected_ids),'participants':len(selected_ids),'resolved_outcome_participants':len(selected_ids&labeled),
                              'participants_lost_from_previous':len(previous-selected_ids)})
            waterfall.append({'policy':policy,'modality':'MRI' if t==MRI else 'CSF','stage_order':len(stages)+1,'stage':'resolved_outcome',
                              'rows':len(selected_ids&labeled),'participants':len(selected_ids&labeled),'resolved_outcome_participants':len(selected_ids&labeled),
                              'participants_lost_from_previous':len(selected_ids-labeled)})
            for reason,n in Counter(byout[rid]['outcome_reason'] for rid in selected_ids-labeled).items():
                unknown.append({'policy':policy,'modality':'MRI' if t==MRI else 'CSF','outcome_reason':reason,'participants':n})
        # Date-only exports cannot order same-day collection relative to prediction.
        # Quantify the boundary separately, without placing those samples in exploration.
        for rid,o in byout.items():
            b={'EXAMDATE':o['baseline_date'],'PHASE':o['baseline_phase']}
            rr=[r for r in indexes[t].get(rid,[]) if dt(r['RUNDATE'])]
            v,r,reason,n=select_measurement(t,rr,b,cfg,90,'acquisition_by_baseline')
            if r and any(x is not None for x in v.values()):
                same_day.append({'RID':rid,'modality':'MRI' if t==MRI else 'CSF','collection_on_baseline_day':int(r['EXAMDATE']==o['baseline_date']),
                                 'label_provisional':o['label_provisional'],'source_row':r['_row'],'collection_date':r['EXAMDATE'],'processing_date':r['RUNDATE']})
    coverage=[];eligibility=[]
    for policy,rr in datasets.items():
        for definition in ['any','complete']:
            flags={r['RID']:{m:(any if definition=='any' else all)(present(r[f]) for f in fs) for m,fs in modalities.items()} for r in rr}
            for count in range(1,len(modalities)+1):
                for combo in itertools.combinations(modalities,count):
                    ids={rid for rid,m in flags.items() if all(m[c] for c in combo)}
                    known=ids&labeled
                    coverage.append({'policy':policy,'modalities':'+'.join(combo),'definition':definition,'master_available':len(ids),
                                     'resolved_36_month_outcome':len(known),'converters':sum(byout[rid]['label_provisional']=='1' for rid in known),
                                     'nonconverters':sum(byout[rid]['label_provisional']=='0' for rid in known),'unresolved_outcome':len(ids-known)})
                    for rid in sorted(ids):eligibility.append({'policy':policy,'definition':definition,'modalities':'+'.join(combo),'RID':rid,'resolved_outcome':int(rid in known)})
        write_csv(out/(policy+'_baseline_predictors.csv'),rr,header)
    # Linkage is descriptive: the primary workflow does not require Key MRI membership.
    for (policy,modality,status),n in Counter((r['policy'],r['modality'],r['key_mri_link_status']) for r in selection if r['has_usable_values']).items():
        subset=[r for r in selection if r['policy']==policy and r['modality']==modality and r['key_mri_link_status']==status and r['has_usable_values']]
        link_summary.append({'policy':policy,'modality':modality,'link_status':status,'selected_participants':n,'resolved_outcomes':sum(r['RID'] in labeled for r in subset),'used_as_exclusion':0})
    for name,rows in [('coverage',coverage),('attrition_waterfall',waterfall),('overlapping_blockers',blockers),('selected_outcome_gaps',unknown),('linkage_summary',link_summary),('same_day_boundary_audit',same_day)]:
        write_csv(report/(name+'.csv'),rows)
    write_csv(out/'source_record_audit.csv',audit)
    write_csv(out/'measurement_selection.csv',selection)
    write_csv(out/'participant_eligibility.csv',eligibility)
    # Outcome audit remains byte-identical to the primary labels; no label recomputation.
    (out/'outcome_audit.csv').write_bytes((primary/'outcome_audit.csv').read_bytes())
    (out/'predictor_schema.json').write_bytes((primary/'predictor_schema.json').read_bytes())
    checks=[]
    def verify(name,ok):
        checks.append({'check':name,'passed':bool(ok)})
        if not ok:raise ValueError(name)
    verify('primary_files_and_raw_inputs_unchanged',all(digest(ROOT/p)==h for p,h in before.items()))
    verify('one_row_per_participant_each_policy',all(len(rr)==len(byout)==len({r['RID'] for r in rr}) for rr in datasets.values()))
    verify('labels_unchanged',digest(out/'outcome_audit.csv')==digest(primary/'outcome_audit.csv'))
    verify('no_after_baseline_collection_selected',all(r['collection_offset_days']=='' or -90<=r['collection_offset_days']<=0 for r in selection))
    verify('exploration_strictly_before_prediction_day',all(r['policy']=='strict' or r['collection_offset_days']=='' or r['collection_offset_days']<0 for r in selection))
    verify('separate_known_processing_dates',all(not r['source_row'] or (dt(r['processing_date']) and dt(r['processing_date'])>=dt(r['collection_date'])) for r in selection))
    verify('strict_processing_by_baseline',all(r['policy']!='strict' or not r['source_row'] or dt(r['processing_date'])<=dt(r['baseline_date']) for r in selection))
    verify('clinical_and_APOE_unchanged',all(r[f]==primary_rows[i][f] for rr in datasets.values() for i,r in enumerate(rr) for f in modalities['clinical']+modalities['APOE']))
    _,old_coverage=read_csv(primary_reports/'modality_coverage.csv')
    verify('strict_counts_reproduce_primary',all(any(r['policy']=='strict' and r['modalities']==old['modalities'] and r['definition']==old['definition'] and r['resolved_36_month_outcome']==int(old['eligible_labeled']) and r['master_available']==int(old['master_available']) for r in coverage) for old in old_coverage))
    verify('all_15_combinations_both_definitions_both_policies',len(coverage)==60)
    write_csv(report/'integrity_checks.csv',checks)
    (report/'manifest.json').write_text(json.dumps({'settings':settings,'primary_config':cfg,'primary_file_hashes':before,
        'script_sha256':digest(Path(__file__)),'sensitivity_config_sha256':digest(local_path(config)),
        'master_participants':len(byout),'resolved_outcomes':len(labeled),'checks_passed':len(checks)},indent=2)+'\n')
    lines=['# MRI and CSF availability sensitivity\n',
           'Primary files are unchanged. No models were trained. Cohort and provisional 36-month labels are frozen from the primary run.\n',
           f'Master: {len(byout)} participants; resolved outcomes: {len(labeled)}.\n',
           '## Policies\n',
           '- **Strict:** reproduce the existing 90-day, inclusive [-90, 0] collection window and processing on/before baseline.',
           '- **Retrospective processing (exploratory):** collection in [-90, -1] days, with known processing date on/after collection; processing after baseline is permitted. No future scan/sample is used. Same-day collections are omitted because date-only records cannot prove collection before prediction start. This boundary difference is explicitly quantified separately.',
           '- Clinical, APOE, baseline/outcome rules, QC, assay verification, values, phase/visit matching and tie resolution are unchanged. Only MRI/CSF processing availability is relaxed, with the additional conservative same-day restriction.',
           '- Processing may occur after the outcome horizon in exploration; this is retrospective measurement of an earlier specimen/scan, not proof that a prediction was operationally possible at baseline. Neither outcomes nor follow-up are used to select measurements.',
           '- RUNDATE is a processing/experiment date, not a verified release or QC-finalization timestamp. EXAMDATE is retained separately as collection/acquisition date.\n',
           '## Resolved-outcome counts\n',
           'Any = at least one selected feature in each modality. Complete = every selected feature in each modality, including CSF ratio. Clinical only means a feature-set comparison, not exclusion of participants who also have other modalities.\n',
           '|Modalities|Strict any|Exploratory any|Strict complete|Exploratory complete|',
           '|---|---:|---:|---:|---:|']
    for combo in dict.fromkeys(r['modalities'] for r in coverage):
        vals=[next(r['resolved_36_month_outcome'] for r in coverage if r['modalities']==combo and r['policy']==p and r['definition']==d) for d in ['any','complete'] for p in policies]
        lines.append('|'+combo+'|'+'|'.join(map(str,vals))+'|')
    lines+=['\nClass counts and unresolved-outcome counts are in coverage.csv. The master keeps participants with missing modalities.\n',
            '## Why records are unavailable\n',
            'The sequential waterfall reports rows, unique participants, and already-resolved participants after each gate. Its losses depend on gate order. overlapping_blockers.csv instead counts overlapping reasons among same-phase initial-visit candidates; these counts must not be added. Source audit includes all MRI/CSF records for master participants, including later visits.\n']
    def gate(policy,modality,stage):
        return next(r for r in waterfall if r['policy']==policy and r['modality']==modality and r['stage']==stage)
    for modality in ['MRI','CSF']:
        s=gate('strict',modality,'collection_window')
        p=gate('strict',modality,'processed_by_baseline')
        e=gate('retrospective_processing',modality,'selected_record_with_usable_values')
        missing_collection=next(r for r in blockers if r['policy']=='strict' and r['modality']==modality and r['reason']=='missing_or_invalid_collection_date')
        missing_processing=next(r for r in blockers if r['policy']=='strict' and r['modality']==modality and r['reason']=='missing_or_invalid_processing_date')
        lines.append(f"- **{modality}, dates:** initial-visit candidates have {missing_collection['rows']} missing/invalid collection dates and {missing_processing['rows']} missing/invalid processing dates. Of {s['participants']} participants with collection inside the primary window ({s['resolved_outcome_participants']} labeled), requiring processing by baseline leaves {p['participants']} ({p['resolved_outcome_participants']} labeled), before downstream QC/batch restrictions.")
        lines.append(f"- **{modality}, follow-up after exploratory selection:** {e['participants']} have usable selected measurements, but only {e['resolved_outcome_participants']} have resolved outcomes. Missing or uncertain outcomes therefore remove {e['participants']-e['resolved_outcome_participants']} from labeled comparisons.")
    q0=gate('retrospective_processing','MRI','processing_on_or_after_collection')
    q1=gate('retrospective_processing','MRI','QC_requirement')
    b0=gate('retrospective_processing','CSF','QC_requirement')
    b1=gate('retrospective_processing','CSF','verified_assay_batch')
    lines.append(f"- **MRI QC:** {q0['participants']} participants remain before QC and {q1['participants']} after finalized overall/hippocampal Pass. One further issue may be conflicting latest scans; see the selection reasons. The eligible MRI class distribution is reported in coverage.csv; a subset with only converters cannot support binary classification evaluation.")
    lines.append(f"- **CSF assay documentation:** {b0['participants']} participants remain before batch verification and {b1['participants']} after it. This is a documentation restriction, not evidence that excluded batches have biologically invalid CSF.")
    qc_counts=Counter((r['STATUS'],r['OVERALLQC'],r['HIPPOQC']) for r in audit if r['modality']=='MRI' and r['phase_and_visit'] and r['collection_strictly_before_baseline'])
    write_csv(report/'MRI_QC_candidate_categories.csv',[
        {'STATUS':status,'OVERALLQC':overall,'HIPPOQC':hippo,'candidate_rows':n}
        for (status,overall,hippo),n in sorted(qc_counts.items())])
    lines.append('MRI_QC_candidate_categories.csv distinguishes unfinalized/missing QC from partial or failed regional ratings; its counts are scan rows, not unique participants.\n')
    for t in ['MRI','CSF']:
        lines.append('### '+t+'\n')
        lines+=['|Policy|Gate|Participants|With resolved outcome|','|---|---|---:|---:|']
        for r in waterfall:
            if r['modality']==t:lines.append(f"|{r['policy']}|{r['stage']}|{r['participants']}|{r['resolved_outcome_participants']}|")
    lines+=['\nMissing Key MRI matches are **not an exclusion criterion** in either policy; they cannot explain removal from this dataset. linkage_summary.csv shows their effect if metadata linkage were later required. MRI QC remains complete/overall Pass/hippocampal Pass. CSF retains only verified batch 12; unverified batches remain withheld, independently of timing.\n',
            'For participants with usable selected measurements but unresolved outcomes, selected_outcome_gaps.csv distinguishes insufficient follow-up, late AD without supporting prior follow-up, and other diagnostic uncertainty. Missing follow-up is not recoded to non-conversion.\n',
            '## Same-day boundary diagnostic (not eligible in exploration)\n']
    for t in ['MRI','CSF']:
        rr=[r for r in same_day if r['modality']==t]
        lines.append(f"- {t}: permitting the primary inclusive baseline-day collection boundary with retrospective processing gives {len(rr)} available participants, {sum(r['label_provisional'] in {'0','1'} for r in rr)} resolved; {sum(r['collection_on_baseline_day'] for r in rr)} selected records are baseline-day collections.")
    lines+=['\n## Reproduce and inspect\n',
            'From the project directory: `python3 scripts/analyze_mri_csf_sensitivity.py`. No extra dependencies, network access or browser required. Optional `--config` selects a separate sensitivity configuration.\n',
            'Outputs: data/sensitivity/mri_csf contains the two predictor snapshots, unchanged outcomes, source-record audit, selection provenance and eligibility. reports/sensitivity/mri_csf contains counts, waterfalls, overlapping blockers, linkage, outcome gaps, integrity checks and hashed manifest. Existing clinical/assay definitions and sources are in docs/decisions.md and docs/sources.md.\n',
            'These are feasibility counts, not evidence of model performance or adequate statistical power. QC and assay-method verification were not relaxed. Review the exploratory availability assumption and batch documentation before model training.\n']
    (report/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'master':len(byout),'resolved':len(labeled),'integrity_checks':len(checks),'report':str((report/'README.md').relative_to(ROOT))},indent=2))
    for r in coverage:
        if r['definition']=='any' and r['modalities'] in {'clinical','MRI','CSF','clinical+APOE','clinical+MRI','clinical+CSF','clinical+MRI+CSF+APOE'}:
            print(r)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/mri_csf_sensitivity.json')
    main(parser.parse_args().config)
