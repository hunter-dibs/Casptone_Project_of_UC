#!/usr/bin/env python3
"""Local-only ADNI preparation. No network, models, imputation or feature fitting."""
import argparse
import calendar
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
TABLES = ['PTDEMOG', 'DXSUM', 'MMSE', 'ADAS', 'NEUROBAT', 'FAQ', 'Key_MRI',
          'UCSFFSX7', 'UPENNBIOMK_ROCHE_ELECSYS', 'APOERES']
CSF = 'UPENNBIOMK_ROCHE_ELECSYS'
MRI = 'UCSFFSX7'
FIELDS = {
 'PTDEMOG': ['PTGENDER', 'PTEDUCAT'], 'MMSE': ['MMSCORE'], 'ADAS': ['TOTAL13'],
 'NEUROBAT': ['AVTOT1','AVTOT2','AVTOT3','AVTOT4','AVTOT5','AVTOT6','AVTOTB',
              'AVDEL30MIN','AVDELTOT','LIMMTOTAL','LDELTOTAL'],
 'FAQ': ['FAQTOTAL'], MRI: ['ST29SV','ST88SV','ST10CV','ST24CV','ST83CV','ST24TA','ST83TA'],
 CSF: ['ABETA42','ABETA40','TAU','PTAU'], 'APOERES': ['GENOTYPE']}
BL_CODES = {'bl','sc','scmri'}
MISSING = {'','NA','N/A','NaN','nan','NULL','-4','-1','-4.0','-1.0'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = [{k: (v or '').strip() for k, v in r.items()} for r in reader]
    return header, rows


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def dt(value):
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def add_months(d, months):
    n = d.year * 12 + d.month - 1 + months
    y, m = divmod(n, 12)
    return date(y, m+1, min(d.day, calendar.monthrange(y, m+1)[1]))


def num(value):
    if value in MISSING or value is None:
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) and n >= 0 else None
    except ValueError:
        return None


def local_path(value):
    from dataset_paths import project_path
    return project_path(value)


def table_name(name):
    if name.upper().startswith('DATADIC') and name.lower().endswith('.csv'):
        return 'DATADIC'
    for t in TABLES:
        if re.search(r'(?:^|_)'+re.escape(t)+r'(?:_|\.)', name, re.I) and name.lower().endswith('.csv'):
            return t
    return None


def organize(raw):
    """Copy originals; extract only recognized CSVs without trusting archive paths."""
    discovery = []
    candidates = [(p, None) for p in sorted(set(ROOT.glob('*.csv')) | set(raw.rglob('*.csv')))]
    for zpath in sorted(p for p in ROOT.iterdir() if p.suffix.lower()=='.zip'):
        discovery.append({'path':str(zpath.relative_to(ROOT)), 'type':'archive', 'sha256':digest(zpath)})
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                p = Path(info.filename)
                if p.is_absolute() or '..' in p.parts or '\\' in info.filename or ((info.external_attr>>16)&0o170000)==0o120000:
                    raise ValueError('Unsafe ZIP member: '+info.filename)
                if info.file_size > 1024**3:
                    raise ValueError('ZIP member exceeds 1 GiB limit')
                if table_name(p.name):
                    dest = raw / ('dictionaries' if table_name(p.name)=='DATADIC' else 'tables') / p.name
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    content = z.read(info)
                    if dest.exists() and hashlib.sha256(content).hexdigest()!=digest(dest):
                        raise ValueError('Conflicting archive file: '+str(dest))
                    if not dest.exists(): dest.write_bytes(content)
    for p, _ in candidates:
        t = table_name(p.name)
        if not t: continue
        dest = raw / ('dictionaries' if t=='DATADIC' else 'tables') / p.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and digest(p)!=digest(dest):
            raise ValueError('Raw copy differs from original: '+str(p))
        if not dest.exists(): shutil.copy2(p,dest)
        discovery.append({'path':str(p.relative_to(ROOT)), 'type':'original_csv', 'sha256':digest(p)})
    return discovery


def group(rows, key='RID'):
    out = defaultdict(list)
    for r in rows: out[r.get(key,'')].append(r)
    return out


def diagnosis(r):
    # Current harmonized coding, verified against dictionary mapping and official guide.
    d = r.get('DIAGNOSIS','')
    if d=='1': return 'CN'
    if d=='2': return 'MCI'
    if d!='3': return 'UNKNOWN'
    if r['PHASE']=='ADNI1':
        return 'AD' if r.get('DXAD')=='1' else 'DEMENTIA_UNSPECIFIED'
    if r.get('DXDDUE')=='1': return 'AD'
    if r.get('DXDDUE')=='2': return 'OTHER_DEMENTIA'
    return 'DEMENTIA_UNSPECIFIED'


def dx_bad(r):
    return r.get('DXCONFID')=='1' or r.get('HAS_QC_ERROR') not in ('','0')


def make_cohort(data, bad_ids, cfg):
    baseline, outcomes, exclusions, events = {}, [], [], []
    for rid, rows in sorted(group(data['DXSUM']).items()):
        reason = ''
        bls = [r for r in rows if r['VISCODE2']=='bl']
        dated = [r for r in bls if dt(r['EXAMDATE'])]
        if rid in bad_ids or not rid: reason='identifier_conflict_or_missing'
        elif not bls: reason='no_explicit_study_baseline'
        elif len(dated)!=len(bls): reason='baseline_date_missing'
        elif len({r['EXAMDATE'] for r in dated})!=1: reason='multiple_explicit_baseline_dates'
        else:
            b = sorted(dated,key=lambda r:r['_row'])[0]
            bd = dt(b['EXAMDATE'])
            same = [r for r in rows if r['EXAMDATE']==b['EXAMDATE']]
            if len({diagnosis(r) for r in same})!=1: reason='baseline_diagnosis_conflict'
            elif diagnosis(b)!='MCI': reason='baseline_not_MCI'
            elif any(dx_bad(r) for r in same): reason='baseline_uncertain_or_qc'
            elif any(dt(r['EXAMDATE']) and dt(r['EXAMDATE'])<bd and r['VISCODE2'] not in {'sc','scmri','bl'} for r in rows):
                reason='followup_before_explicit_baseline'
        if reason:
            exclusions.append({'RID':rid,'reason':reason,'explicit_baseline_rows':len(bls)})
            continue
        baseline[rid] = b
        horizon = add_months(bd, cfg['horizon_months'])
        threshold = horizon-timedelta(days=cfg['followup_tolerance_days'])
        timeline = defaultdict(list)
        for r in rows:
            d = dt(r['EXAMDATE'])
            if d and d>bd: timeline[d].append(r)
        seq=[]
        for d, rr in sorted(timeline.items()):
            states={diagnosis(r) for r in rr}
            conflict=len(states)>1
            state=next(iter(states)) if not conflict else 'CONFLICT'
            uncertain=conflict or any(dx_bad(r) for r in rr) or state in {'UNKNOWN','DEMENTIA_UNSPECIFIED'}
            seq.append((d,state,uncertain))
        valid_ad=[d for d,s,u in seq if s=='AD' and not u]
        first_ad=min(valid_ad,default=None)
        # Event is first documented AD, never inferred onset between visits.
        conversion=first_ad is not None and first_ad<=horizon
        support=next((d for d,s,u in seq if d>=threshold and s in {'CN','MCI'} and not u),None)
        end=first_ad if conversion else support or horizon
        unresolved=any(u or s=='OTHER_DEMENTIA' for d,s,u in seq if d<=end)
        undated=any(not dt(r['EXAMDATE']) for r in rows)
        prior_ad=any(dt(r['EXAMDATE']) and dt(r['EXAMDATE'])<bd and diagnosis(r) in {'AD','OTHER_DEMENTIA','DEMENTIA_UNSPECIFIED'} for r in rows)
        ad_reversion=bool(first_ad and any(d>first_ad and s in {'CN','MCI'} for d,s,u in seq))
        cn_reversion=any(s=='CN' and d<=horizon for d,s,u in seq)
        if prior_ad: label,reason='', 'prebaseline_dementia'
        elif unresolved or undated: label,reason='', 'uncertain_conflicting_or_undated_diagnosis'
        elif conversion: label,reason=1,'AD_documented_within_horizon'
        elif support and (first_ad is None or first_ad>support): label,reason=0,'non_AD_followup_at_or_after_threshold'
        elif first_ad and first_ad>horizon: label,reason='', 'AD_after_horizon_without_prior_support'
        else: label,reason='', 'insufficient_followup'
        last=max(timeline,default=bd)
        o={'RID':rid,'PTID':b['PTID'],'baseline_date':bd.isoformat(),'baseline_phase':b['PHASE'],
           'baseline_source_row':b['_row'],'horizon_date':horizon.isoformat(),'horizon_months':cfg['horizon_months'],
           'label_provisional':label,'outcome_reason':reason,'first_AD_date':first_ad.isoformat() if first_ad else '',
           'support_date':support.isoformat() if support else '', 'last_diagnosis_date':last.isoformat(),
           'followup_days':(last-bd).days,'diagnosis_followup_dates':len(seq),
           'any_uncertain_diagnosis':int(any(u for d,s,u in seq)), 'undated_diagnosis':int(undated),
           'any_conflict':int(any(s=='CONFLICT' for d,s,u in seq)),
           'MCI_to_CN_within_horizon':int(cn_reversion),'AD_reversion_any_followup':int(ad_reversion),
           'prebaseline_dementia':int(prior_ad),
           'strict_no_reversion_eligible':int(label!='' and not cn_reversion and not ad_reversion)}
        outcomes.append(o)
        for r in rows:
            d=dt(r['EXAMDATE'])
            events.append({k:r.get(k,'') for k in ['RID','PTID','PHASE','VISCODE','VISCODE2','EXAMDATE','DIAGNOSIS','DXAD','DXDDUE','DXCONFID','HAS_QC_ERROR','_file','_row']} |
                          {'interpreted_diagnosis':diagnosis(r),'offset_days':(d-bd).days if d else '', 'uncertain_or_qc':int(dx_bad(r))})
    return baseline,outcomes,exclusions,events


def age(r, baseline_date):
    s=r.get('PTDOB','')
    if re.fullmatch(r'\d{2}/\d{4}',s):
        month,year=map(int,s.split('/'))
        try: birth=date(year,month,15)
        except ValueError: return None,'invalid_birth_month'
        method='month_known_day_15_approximation'
    elif dt(s): birth=dt(s); method='recorded_full_date_precision_unverified'
    elif re.match(r'^\d{4}(?:-01-01)?$',r.get('PTDOBYY','')):
        birth=date(int(r['PTDOBYY'][:4]),7,1);method='year_only_July_1_approximation'
    else: return None,'birth_date_missing'
    if birth>baseline_date:return None,'birth_after_baseline'
    return round((baseline_date-birth).days/365.2425,4),method


def cleaned(t,r,cfg):
    values={f: num(r.get(f,'')) for f in FIELDS[t] if f!='GENOTYPE'}
    flags=[]
    for f,n in list(values.items()):
        upper=30 if f in {'MMSCORE','FAQTOTAL'} else 85 if f=='TOTAL13' else 25 if f in {'LIMMTOTAL','LDELTOTAL'} else 15 if f.startswith('AV') else None
        if (upper is not None and n is not None and n>upper) or (f=='PTGENDER' and n not in {None,1,2}):
            values[f]=None;flags.append(f+':invalid_range')
        if t==MRI and n is not None and n<=0:values[f]=None;flags.append(f+':nonpositive')
    if r.get('HAS_QC_ERROR') not in (None,'','0'):
        values={f:None for f in values};flags.append('CRF_QC_error')
    if t==CSF:
        comment=r.get('COMMENT','').lower()
        if r.get('BATCH') not in cfg['csf_verified_batches']:
            values={f:None for f in values};flags.append('batch_limits_unverified')
        for f,n in list(values.items()):
            lo,hi=cfg['csf_limits'][f]
            if n is not None and not lo<=n<=hi:
                values[f]=None;flags.append(f+':outside_technical_range')
        # Free-text specimen problems and inequality annotations are not exact measurements.
        if 'hemoly' in comment or 'failed' in comment:
            values={f:None for f in values};flags.append('specimen_or_recalculation_problem')
        for match in re.finditer(r'(abeta42|abeta40|ptau|tau)\s*[<>]',comment):
            f=match.group(1).upper();values[f]=None;flags.append(f+':censored_comment')
        a,b=values['ABETA42'],values['ABETA40']
        values['ABETA42_ABETA40_RATIO']=a/b if a is not None and b is not None and b>0 else None
    return values,flags


def mri_qc(r):
    return r.get('STATUS','').lower()=='complete' and r.get('OVERALLQC','').lower()=='pass' and r.get('HIPPOQC','').lower()=='pass'


def date_field(t):
    return 'EXAMDATE' if t in {MRI,CSF} else 'VISDATE'


def select_measurement(t,rows,b,cfg,lookback=None,policy=None):
    bd=dt(b['EXAMDATE']); window=cfg['lookback_days'][t] if lookback is None else lookback
    policy=policy or cfg['availability_policy']
    candidates=[]
    for r in rows:
        d=dt(r.get(date_field(t),''))
        if r.get('PHASE')!=b['PHASE'] or r.get('VISCODE2') not in BL_CODES or not d:continue
        if not -window<=(d-bd).days<=0:continue
        if t==MRI and not mri_qc(r):continue
        if r.get('HAS_QC_ERROR') not in (None,'','0'):continue
        run=dt(r.get('RUNDATE',''))
        if t in {MRI,CSF}:
            if run and run<d:continue
            if policy=='result_by_baseline' and (not run or run>bd):continue
        candidates.append(r)
    if not candidates:return {},None,'no_eligible_record',0
    latest=max(r[date_field(t)] for r in candidates)
    tied=[r for r in candidates if r[date_field(t)]==latest]
    compare=FIELDS[t]+(['PTDOB','PTDOBYY'] if t=='PTDEMOG' else ['BATCH'] if t==CSF else ['IMAGEUID'] if t==MRI else [])
    if len({tuple(r.get(f,'') for f in compare) for r in tied})>1:
        return {},None,'conflicting_latest_records',len(candidates)
    selected=min(tied,key=lambda r:r['_row'])
    values,flags=cleaned(t,selected,cfg)
    if t=='PTDEMOG':values['AGE_APPROX'],method=age(selected,bd);flags.append(method)
    return values,selected,';'.join(flags) or 'selected',len(candidates)


def apoe(rows,b,cfg):
    valid=[r for r in rows if re.fullmatch('[234]/[234]',r.get('GENOTYPE','')) and r.get('APUSABLE') in (None,'','1')]
    if len({r['GENOTYPE'] for r in valid})>1:return {},None,'conflicting_genotypes',len(valid)
    if cfg['apoe_policy']=='tested_by_baseline':
        valid=[r for r in valid if dt(r.get('APTESTDT','')) and dt(r['APTESTDT'])<=dt(b['EXAMDATE'])]
    if not valid:return {},None,'no_genotype_with_verified_availability',0
    r=min(valid,key=lambda r:(r.get('APTESTDT') or '9999',r['_row']))
    return {'E4_COUNT':r['GENOTYPE'].split('/').count('4')},r,'selected',len(valid)


def logical_memory(rows,b,cfg,lookback=None,policy=None):
    # Logical Memory is usually recorded at screening, RAVLT at baseline.
    # Select the two LM scores together from one independently dated record.
    eligible=[r for r in rows if any(num(r.get(f,'')) is not None for f in ['LIMMTOTAL','LDELTOTAL'])]
    v,r,reason,n=select_measurement('NEUROBAT',eligible,b,cfg,lookback,policy)
    return {f:v.get(f) for f in ['LIMMTOTAL','LDELTOTAL']},r,reason,n


def main(config='config/workflow.json'):
    cfg=json.loads(local_path(config).read_text())
    if not isinstance(cfg['horizon_months'],int) or cfg['horizon_months']<=0:raise ValueError('Positive integer horizon required')
    if cfg['followup_tolerance_days']<0 or cfg['followup_tolerance_days']>=cfg['horizon_months']*28:raise ValueError('Invalid follow-up tolerance')
    if any(not isinstance(x,int) or x<0 for x in cfg['lookback_days'].values()):raise ValueError('Lookbacks must be nonnegative integer days')
    if cfg['availability_policy'] not in {'result_by_baseline','acquisition_by_baseline'}:raise ValueError('Unknown availability policy')
    if cfg['apoe_policy'] not in {'tested_by_baseline','time_invariant_research'}:raise ValueError('Unknown APOE policy')
    raw,out,report=(local_path(cfg[k]) for k in ['raw_dir','output_dir','report_dir'])
    for p in [raw,out,report]:p.mkdir(parents=True,exist_ok=True)
    discovery=organize(raw)
    write_csv(report/'discovery.csv',discovery)
    files={}
    for t in TABLES+['DATADIC']:
        matches=[p for p in raw.rglob('*.csv') if table_name(p.name)==t]
        if len(matches)!=1:raise ValueError(f'Expected one {t} file under {raw}; found {len(matches)}. Supply missing file or resolve versions.')
        files[t]=matches[0]
    # Load and retain supplied definitions BEFORE deriving variables.
    _,dictionary=read_csv(files['DATADIC'])
    relevant=[r for r in dictionary if r['TBLNAME'] in TABLES]
    write_csv(report/'dictionary_extract.csv',relevant)
    dictionary_keys={(r['TBLNAME'],r['FLDNAME']) for r in relevant}
    for t,fields in FIELDS.items():
        for f in fields:
            if (t,f) not in dictionary_keys:raise ValueError(f'No dictionary definition for {t}.{f}')
    data={}; inventory=[]; quality=[]; missingness=[]; visit_counts=[]; conflicts=[]
    rid_ptids=defaultdict(set);ptid_rids=defaultdict(set)
    for t,p in files.items():
        h,rows=read_csv(p)
        inventory.append({'table':t,'filename':str(p.relative_to(ROOT)),'sha256':digest(p),'bytes':p.stat().st_size,
                          'rows':len(rows),'columns':len(h),'unique_RID':len({r['RID'] for r in rows if r.get('RID')}) if 'RID' in h else '',
                          'unique_PTID':len({r.get('PTID',r.get('subject_id')) for r in rows if r.get('PTID',r.get('subject_id'))}) if t!='DATADIC' else ''})
        if t=='DATADIC':continue
        for f in FIELDS.get(t,[]):
            if f not in h:raise ValueError(f'Required predictor column missing: {t}.{f}')
        exact=len(rows)-len({tuple(r.values()) for r in rows})
        # Retain every source row; selection only collapses substantively identical ties.
        for i,r in enumerate(rows,2):
            r['_row']=i;r['_file']=str(p.relative_to(ROOT))
            if r.get('RID') and r.get('PTID'):
                rid_ptids[r['RID']].add(r['PTID']);ptid_rids[r['PTID']].add(r['RID'])
        data[t]=rows
        keys=['image_id'] if t=='Key_MRI' else ['RID'] if t=='APOERES' else ['RID','PHASE','VISCODE2', 'IMAGEUID'] if t==MRI else ['RID','PHASE','VISCODE2','EXAMDATE','BATCH'] if t==CSF else ['RID','PHASE','VISCODE2']
        groups=defaultdict(list)
        for r in rows:groups[tuple(r.get(k,'') for k in keys)].append(r)
        conflict_groups=0
        for key,rr in groups.items():
            if len(rr)<2:continue
            cf=[f for f in h if f not in {'ID','USERDATE','USERDATE2','update_stamp'} and len({r[f] for r in rr})>1]
            if cf:
                conflict_groups+=1
                conflicts.append({'table':t,'key_fields':'|'.join(keys),'key':'|'.join(key),'rows':'|'.join(str(r['_row']) for r in rr),'conflicting_fields':'|'.join(cf)})
        quality.append({'table':t,'exact_duplicate_rows':exact,'record_key':'|'.join(keys),'duplicate_key_groups':sum(len(rr)>1 for rr in groups.values()),'conflicting_key_groups':conflict_groups})
        for f in h:
            c=Counter(r[f] for r in rows)
            missingness.append({'table':t,'column':f,'rows':len(rows),'missing_or_sentinel':sum(c[x] for x in MISSING),'missing_fraction':round(sum(c[x] for x in MISSING)/len(rows),6),'minus4':c['-4']+c['-4.0'],'minus1':c['-1']+c['-1.0']})
        for key,n in Counter((r.get('PHASE',''),r.get('VISCODE',''),r.get('VISCODE2','')) for r in rows).items():
            visit_counts.append({'table':t,'PHASE':key[0],'VISCODE':key[1],'VISCODE2':key[2],'rows':n})
    bad_ids={rid for rid,ps in rid_ptids.items() if len(ps)!=1}
    for p,rs in ptid_rids.items():
        if len(rs)!=1:bad_ids.update(rs)
    crosswalk=[{'RID':rid,'PTID':'|'.join(sorted(ps)),'consistent':int(rid not in bad_ids)} for rid,ps in sorted(rid_ptids.items())]
    for name,rows in [('inventory',inventory),('table_quality',quality),('raw_missingness',missingness),('record_conflicts',conflicts),('visit_code_audit',visit_counts),('identifier_crosswalk',crosswalk)]:write_csv(report/(name+'.csv'),rows)
    baseline,outcomes,exclusions,events=make_cohort(data,bad_ids,cfg)
    write_csv(report/'cohort_exclusions.csv',exclusions)
    write_csv(out/'diagnosis_event_audit.csv',events)
    indexes={t:group(data[t]) for t in FIELDS}
    features={t:[t+'__'+f for f in fs if f!='GENOTYPE'] for t,fs in FIELDS.items()}
    features['PTDEMOG'].append('PTDEMOG__AGE_APPROX')
    features[CSF].append(CSF+'__ABETA42_ABETA40_RATIO')
    features['APOERES']=['APOERES__E4_COUNT']
    feature_list=[f for fs in features.values() for f in fs]
    master=[]; provenance=[]; timing=[]; candidates=[]; key_audit=[]
    key_images=group(data['Key_MRI'],'image_id')
    for rid,b in sorted(baseline.items()):
        bd=dt(b['EXAMDATE']); result={'RID':rid,'PTID':b['PTID']}|{f:None for f in feature_list}
        for t in FIELDS:
            rows=indexes[t].get(rid,[])
            values,r,reason,n=apoe(rows,b,cfg) if t=='APOERES' else select_measurement(t,rows,b,cfg)
            result.update({t+'__'+f:v for f,v in values.items()})
            d=dt(r.get('APTESTDT' if t=='APOERES' else date_field(t),'')) if r else None
            provenance.append({'RID':rid,'table':t,'reason':reason,'candidate_count':n,'source_file':r['_file'] if r else '',
                               'source_row':r['_row'] if r else '', 'source_phase':r.get('PHASE','') if r else '',
                               'VISCODE':r.get('VISCODE','') if r else '', 'VISCODE2':r.get('VISCODE2','') if r else '',
                               'source_date':d.isoformat() if d else '', 'offset_days':(d-bd).days if d else '',
                               'RUNDATE':r.get('RUNDATE','') if r else '', 'BATCH':r.get('BATCH','') if r else '',
                               'IMAGEUID':r.get('IMAGEUID','') if r else '', 'STATUS':r.get('STATUS','') if r else '',
                               'OVERALLQC':r.get('OVERALLQC','') if r else '', 'HIPPOQC':r.get('HIPPOQC','') if r else '',
                               'COMMENT':r.get('COMMENT','') if r else '', 'availability_policy':cfg['apoe_policy'] if t=='APOERES' else cfg['availability_policy']})
            provenance[-1]['feature_columns']='|'.join(f for f in features[t] if t!='NEUROBAT' or not f.endswith(('LIMMTOTAL','LDELTOTAL')))
            if t=='NEUROBAT':
                lm,lr,lreason,ln=logical_memory(rows,b,cfg)
                result.update({'NEUROBAT__'+f:v for f,v in lm.items()})
                ld=dt(lr.get('VISDATE','')) if lr else None
                lp={k:'' for k in provenance[-1]}
                lp.update({'RID':rid,'table':'NEUROBAT_LOGICAL_MEMORY','reason':lreason,'candidate_count':ln,
                           'source_file':lr['_file'] if lr else '', 'source_row':lr['_row'] if lr else '',
                           'source_phase':lr['PHASE'] if lr else '', 'VISCODE':lr['VISCODE'] if lr else '',
                           'VISCODE2':lr['VISCODE2'] if lr else '', 'source_date':ld.isoformat() if ld else '',
                           'offset_days':(ld-bd).days if ld else '', 'availability_policy':cfg['availability_policy'],
                           'feature_columns':'NEUROBAT__LIMMTOTAL|NEUROBAT__LDELTOTAL'})
                provenance.append(lp)
            if t=='APOERES':continue
            for rr in rows:
                if rr.get('VISCODE2') not in BL_CODES:continue
                dd=dt(rr.get(date_field(t),''));run=dt(rr.get('RUNDATE',''))
                rejected=[]
                if rr.get('PHASE')!=b['PHASE']:rejected.append('different_phase')
                if not dd:rejected.append('missing_measurement_date')
                elif dd>bd:rejected.append('measurement_after_baseline')
                elif (bd-dd).days>cfg['lookback_days'][t]:rejected.append('before_lookback_window')
                if t==MRI and not mri_qc(rr):rejected.append('MRI_QC_not_complete_overall_and_hippocampal_pass')
                if t in {MRI,CSF} and cfg['availability_policy']=='result_by_baseline':
                    if not run:rejected.append('missing_processing_date')
                    elif run>bd:rejected.append('processed_after_baseline')
                    elif dd and run<dd:rejected.append('processing_before_collection')
                if t==CSF and rr['BATCH'] not in cfg['csf_verified_batches']:rejected.append('CSF_batch_methods_unverified')
                candidates.append({'RID':rid,'table':t,'source_row':rr['_row'],'source_date':dd.isoformat() if dd else '',
                                   'VISCODE2':rr.get('VISCODE2',''),'same_phase':int(rr.get('PHASE')==b['PHASE']),
                                   'offset_days':(dd-bd).days if dd else '', 'run_offset_days':(run-bd).days if run else '',
                                   'MRI_QC_pass':int(mri_qc(rr)) if t==MRI else '', 'BATCH':rr.get('BATCH',''),
                                   'rejection_reasons':';'.join(rejected)})
                if t==MRI:
                    matches=key_images.get(rr['IMAGEUID'].removeprefix('I'),[])
                    status='missing_image_uid' if not matches else 'multiple_image_uid_rows' if len(matches)>1 else 'matched' if matches[0]['subject_id']==rr['PTID'] and matches[0]['image_date']==rr['EXAMDATE'] else 'subject_or_date_conflict'
                    km=matches[0] if status=='matched' else {}
                    key_audit.append({'RID':rid,'UCSFFSX7_source_row':rr['_row'],'IMAGEUID':rr['IMAGEUID'],'status':status,'key_source_row':km.get('_row','')} |
                                     {k:km.get(k,'') for k in ['subject_id','image_date','image_visit','series_type','series_description','scanner_manufacturer','scanner_model','magnetic_field_strength','acceleration']})
        master.append(result)
    # Count matched usable measurements under explicit alternative windows; never mix into master.
    for t in FIELDS:
        if t=='APOERES':continue
        for window in cfg['timing_sensitivity_days']:
            for policy in ['result_by_baseline','acquisition_by_baseline']:
                any_n=all_n=0
                for rid,b in baseline.items():
                    v,r,reason,n=select_measurement(t,indexes[t].get(rid,[]),b,cfg,window,policy)
                    if t=='NEUROBAT':v.update(logical_memory(indexes[t].get(rid,[]),b,cfg,window,policy)[0])
                    any_n+=bool(v) and any(x is not None for x in v.values())
                    all_n+=bool(v) and all(x is not None for x in v.values())
                timing.append({'table':t,'lookback_days':window,'availability_policy':policy,'participants_any_selected_feature':any_n,'participants_all_selected_features':all_n})
    write_csv(report/'measurement_candidates.csv',candidates)
    write_csv(report/'measurement_exclusion_summary.csv',[
        {'table':t,'reason':reason,'candidate_rows':n}
        for (t,reason),n in sorted(Counter((r['table'],reason) for r in candidates for reason in r['rejection_reasons'].split(';') if reason).items())])
    write_csv(report/'timing_sensitivity.csv',timing)
    write_csv(out/'key_mri_link_audit.csv',key_audit)
    write_csv(out/'measurement_provenance.csv',provenance)
    write_csv(out/'outcome_audit.csv',outcomes)
    write_csv(out/'baseline_predictors.csv',master,['RID','PTID']+feature_list)
    modalities={'clinical':features['PTDEMOG']+features['MMSE']+features['ADAS']+features['NEUROBAT']+features['FAQ'],
                'MRI':features[MRI], 'CSF':features[CSF], 'APOE':features['APOERES']}
    byout={r['RID']:r for r in outcomes}
    eligibility=[];coverage=[]
    for r in master:
        e={'RID':r['RID'],'label_available':int(byout[r['RID']]['label_provisional']!='')}
        for m,fs in modalities.items():
            e['available_any_'+m]=int(any(r[f] is not None for f in fs))
            e['complete_'+m]=int(all(r[f] is not None for f in fs))
        for count in range(1,5):
            for combo in itertools.combinations(modalities,count):
                e['eligible_any_'+'_'.join(combo)]=int(e['label_available'] and all(e['available_any_'+m] for m in combo))
                e['eligible_complete_'+'_'.join(combo)]=int(e['label_available'] and all(e['complete_'+m] for m in combo))
        eligibility.append(e)
    for count in range(1,5):
        for combo in itertools.combinations(modalities,count):
            for definition in ['any','complete']:
                col='eligible_'+definition+'_'+'_'.join(combo)
                eligible=[e for e in eligibility if e[col]]
                coverage.append({'modalities':'+'.join(combo),'definition':definition,'eligible_labeled':len(eligible),
                                 'converters':sum(byout[e['RID']]['label_provisional']==1 for e in eligible),
                                 'nonconverters':sum(byout[e['RID']]['label_provisional']==0 for e in eligible),
                                 'master_available':sum(all(e[('available_any_' if definition=='any' else 'complete_')+m] for m in combo) for e in eligibility)})
    write_csv(out/'modality_eligibility.csv',eligibility)
    write_csv(report/'modality_coverage.csv',coverage)
    write_csv(report/'predictor_missingness.csv',[{'feature':f,'participants':len(master),'observed':sum(r[f] is not None for r in master),'missing':sum(r[f] is None for r in master)} for f in feature_list])
    schema={'identifiers':['RID','PTID'],'predictor_columns':feature_list,'modalities':modalities,'excluded_from_X':'All other files and all identifier, outcome, date, batch, source, QC and eligibility columns.'}
    (out/'predictor_schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    mapping=[]
    for r in relevant:
        if r['FLDNAME'] in FIELDS.get(r['TBLNAME'],[]) or r['FLDNAME'] in {'RID','PTID','VISCODE','VISCODE2','VISDATE','EXAMDATE','RUNDATE','DIAGNOSIS','DXAD','DXDDUE','DXCONFID','PTDOB','PTDOBYY','STATUS','OVERALLQC','HIPPOQC','BATCH','GENOTYPE','APTESTDT','APUSABLE'}:
            role='predictor' if r['FLDNAME'] in FIELDS.get(r['TBLNAME'],[]) and r['FLDNAME']!='GENOTYPE' else 'outcome_only' if r['TBLNAME']=='DXSUM' else 'identifier' if r['FLDNAME'] in {'RID','PTID'} else 'provenance_or_derivation_only'
            mapping.append({'source_table':r['TBLNAME'],'source_field':r['FLDNAME'],'phase':r['PHASE'],'meaning':r['TEXT'],'units':r['UNITS'] or 'not specified in dictionary','coding':r['CODE'],'mapping_notes':r['MAPPING_NOTES'],'intended_role':role,'definition_source':str(files['DATADIC'].relative_to(ROOT))})
    for t,f,meaning,units,coding in [('PTDEMOG','AGE_APPROX','Approximate age from recorded birth date and diagnosis baseline','years','month: day 15; year: July 1; days / 365.2425'),(CSF,'ABETA42_ABETA40_RATIO','Same specimen and batch Aβ42 divided by Aβ40','dimensionless','only verified batch and both uncensored within technical ranges'),('APOERES','E4_COUNT','Number of epsilon4 alleles','allele count','count of 4 in validated n/m genotype; 0,1,2')]:
        mapping.append({'source_table':t,'source_field':f,'phase':'derived','meaning':meaning,'units':units,'coding':coding,'mapping_notes':'See docs/decisions.md','intended_role':'predictor','definition_source':'docs/sources.md'})
    for f in data['Key_MRI'][0]:
        if not f.startswith('_'):mapping.append({'source_table':'Key_MRI','source_field':f,'phase':'not supplied','meaning':'Unverified export field; retained verbatim in metadata audit where relevant','units':'unverified','coding':'unverified','mapping_notes':'No Key MRI dictionary supplied. Match image_id to IMAGEUID, require subject and date agreement.','intended_role':'provenance_only','definition_source':'observed schema; docs/sources.md'})
    write_csv(report/'variable_mapping.csv',mapping)
    counts=Counter(r['label_provisional'] for r in outcomes)
    flow=[{'stage':'DXSUM participants','reason':'all','count':len(group(data['DXSUM']))}]
    flow += [{'stage':'excluded before master','reason':k,'count':n} for k,n in sorted(Counter(r['reason'] for r in exclusions).items())]
    flow += [{'stage':'master baseline MCI','reason':'all retained regardless of modalities','count':len(master)}]
    flow += [{'stage':'outcome','reason':k,'count':n} for k,n in sorted(Counter(r['outcome_reason'] for r in outcomes).items())]
    write_csv(report/'cohort_flow.csv',flow)
    frequencies=[]
    for t,fs in {'DXSUM':['PHASE','DIAGNOSIS','DXDDUE','DXAD','DXCONFID'],MRI:['STATUS','OVERALLQC','HIPPOQC'],CSF:['BATCH','COMMENT'],'APOERES':['GENOTYPE','APUSABLE']}.items():
        for f in fs:
            for v,n in sorted(Counter(r[f] for r in data[t]).items()):frequencies.append({'table':t,'field':f,'value':v,'rows':n})
    write_csv(report/'coding_frequencies.csv',frequencies)
    write_csv(report/'csf_value_audit.csv',[{'RID':r['RID'],'source_row':r['_row'],'BATCH':r['BATCH'],'EXAMDATE':r['EXAMDATE'],'RUNDATE':r['RUNDATE'],'COMMENT':r['COMMENT'],**{f:r[f] for f in FIELDS[CSF]},'cleaning_flags':';'.join(cleaned(CSF,r,cfg)[1]),'valid_ratio':cleaned(CSF,r,cfg)[0].get('ABETA42_ABETA40_RATIO')} for r in data[CSF]])
    write_csv(report/'neurobat_measurements.csv',[{'field':f,'present':int(f in data['NEUROBAT'][0]),'raw_valid_rows':sum(num(r.get(f,'')) is not None for r in data['NEUROBAT'])} for f in FIELDS['NEUROBAT']])
    # Invariants on actual outputs, explicitly requested by this workflow task.
    checks=[]
    def check(name,condition):
        checks.append({'check':name,'passed':bool(condition)})
        if not condition:raise AssertionError(name)
    check('unique participant master',len(master)==len({r['RID'] for r in master}))
    check('one outcome and eligibility row per master participant',set(baseline)=={r['RID'] for r in outcomes}=={r['RID'] for r in eligibility})
    check('one provenance row per participant and measurement group',len(provenance)==len(master)*(len(FIELDS)+1) and len({(r['RID'],r['table']) for r in provenance})==len(provenance))
    check('no future predictor measurement',all(r['offset_days']=='' or r['offset_days']<=0 for r in provenance))
    check('result availability policy',cfg['availability_policy']!='result_by_baseline' or all(not r['source_row'] or r['table'] not in {MRI,CSF} or (dt(r['RUNDATE']) and dt(r['RUNDATE'])<=dt(baseline[r['RID']]['EXAMDATE'])) for r in provenance))
    check('positive labels inside horizon',all(o['label_provisional']!=1 or dt(o['baseline_date'])<dt(o['first_AD_date'])<=dt(o['horizon_date']) for o in outcomes))
    check('negative labels have sufficient support',all(o['label_provisional']!=0 or (dt(o['support_date']) and dt(o['support_date'])>=dt(o['horizon_date'])-timedelta(days=cfg['followup_tolerance_days'])) for o in outcomes))
    check('unknown outcomes never coerced to zero',all(o['outcome_reason'] not in {'insufficient_followup','uncertain_conflicting_or_undated_diagnosis','prebaseline_dementia','AD_after_horizon_without_prior_support'} or o['label_provisional']=='' for o in outcomes))
    check('predictor allowlist excludes outcomes and identifiers',all(f not in {'RID','PTID'} and not f.startswith('DXSUM') for f in feature_list))
    check('cohort flow reconciles',len(master)+len(exclusions)==len(group(data['DXSUM'])))
    check('original file checksums unchanged',all(digest(ROOT/r['path'])==r['sha256'] for r in discovery))
    write_csv(report/'validation_checks.csv',checks)
    summary={'DXSUM_participants':len(group(data['DXSUM'])),
             'explicit_baseline_MCI_before_exclusions':len({r['RID'] for r in data['DXSUM'] if r['VISCODE2']=='bl' and r['DIAGNOSIS']=='2'}),
             'baseline_MCI_master':len(master),'converters':counts[1],'nonconverters':counts[0],'uncertain_outcomes':counts[''],'excluded_before_master':len(exclusions),'identifier_conflict_RIDs':len(bad_ids),'validation_checks_passed':len(checks)}
    write_csv(report/'cohort_by_phase.csv',[
        {'baseline_phase':phase,'master':len(rr),'converters':sum(r['label_provisional']==1 for r in rr),
         'nonconverters':sum(r['label_provisional']==0 for r in rr),'unknown':sum(r['label_provisional']=='' for r in rr)}
        for phase,rr in sorted(group(outcomes,'baseline_phase').items())])
    write_csv(report/'followup_distribution.csv',[
        {'baseline_phase':phase,'participants':len(rr),'minimum_days':min(r['followup_days'] for r in rr),
         'median_days':sorted(r['followup_days'] for r in rr)[len(rr)//2],
         'maximum_days':max(r['followup_days'] for r in rr),
         'no_dated_followup':sum(r['diagnosis_followup_dates']==0 for r in rr)}
        for phase,rr in sorted(group(outcomes,'baseline_phase').items())])
    (report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (report/'run_manifest.json').write_text(json.dumps({'config':cfg,'python':sys.version,'script_sha256':digest(Path(__file__)),'config_sha256':digest(local_path(config)),'inputs':inventory},indent=2)+'\n')
    (report/'data_quality_report.md').write_text('# Actual data preparation results\n\n'+ '\n'.join(f'- {k}: {v}' for k,v in summary.items())+'\n\nAll counts use the provisional configuration in run_manifest.json. The master retains missing modalities and unknown labels remain in outcome_audit.csv.\n\n'+
        'All ten expected tables and DATADIC were found. Original root CSVs were copied, not moved or edited. Discovery records any archives. RAVLT trials, delay and recognition, plus Logical Memory immediate and delayed scores are present. The dictionary extract and variable mapping retain phase-specific definitions.\n\n'+
        'See table_quality.csv and record_conflicts.csv for duplicate-key audits; raw_missingness.csv for every source column; coding_frequencies.csv for diagnosis, QC and batch codes; measurement_candidates.csv and timing_sensitivity.csv for timing; key_mri_link_audit.csv in dataset/processed/original for scan metadata linkage. No undocumented metadata field is used as a predictor.\n\n'+
        'Feasibility is constrained by the configured prebaseline windows, finalized MRI QC, result availability dates, unverified CSF batch methods, and incomplete follow-up. Modality counts and class balance are in modality_coverage.csv. These counts do not establish adequate power or generalization. Review docs/decisions.md before changing configuration or training.\n\n'+
        '## Current modality coverage\n\n|Modality|Definition|Master available|Labeled eligible|AD|Non-AD|\n|---|---|---:|---:|---:|---:|\n'+
        '\n'.join(f"|{r['modalities']}|{r['definition']}|{r['master_available']}|{r['eligible_labeled']}|{r['converters']}|{r['nonconverters']}|" for r in coverage)+'\n\n'+
        'The user selected a 90-day lookback with results available by baseline on 2026-09-20. The exact active configuration is recorded in run_manifest.json. RUNDATE does not establish release or QC-finalization time. Batch 12 is the only CSF batch with methods verified here; other batches are withheld pending the consolidated methods. The absence of a Key MRI dictionary restricts those metadata to auditing.\n')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/workflow.json')
    main(parser.parse_args().config)
