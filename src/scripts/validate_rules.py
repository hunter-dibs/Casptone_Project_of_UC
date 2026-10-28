#!/usr/bin/env python3
"""Focused synthetic boundary checks, not participant results or model training."""
from copy import deepcopy
from prepare_adni import make_cohort, select_measurement, diagnosis, logical_memory, ROOT
import json


def main():
    cfg=json.loads((ROOT/'config/workflow.json').read_text())
    cfg['horizon_months']=36;cfg['followup_tolerance_days']=0
    cfg['availability_policy']='result_by_baseline'
    cfg['lookback_days']['MMSE']=90
    cfg['lookback_days']['UCSFFSX7']=90
    def row(day,visit='m36',dx='2',cause='',phase='ADNI2'):
        return {'RID':'synthetic','PTID':'synthetic','PHASE':phase,'VISCODE':visit,'VISCODE2':visit,
                'EXAMDATE':day,'DIAGNOSIS':dx,'DXDDUE':cause,'DXAD':'','DXCONFID':'','HAS_QC_ERROR':'0','_row':2,'_file':'synthetic'}
    b=row('2020-01-15','bl')
    def outcome(followups):
        return make_cohort({'DXSUM':[b]+followups},set(),cfg)[1][0]
    assert outcome([row('2023-01-14')])['label_provisional']==''
    assert outcome([row('2023-01-15')])['label_provisional']==0
    assert outcome([row('2023-01-15',dx='3',cause='1')])['label_provisional']==1
    assert outcome([row('2023-01-16',dx='3',cause='1')])['label_provisional']==''
    assert outcome([row('2022-01-15',dx='3')])['label_provisional']==''
    assert outcome([row('2022-01-15',dx='3',cause='2')])['label_provisional']==''
    assert outcome([row('2022-01-15',dx='3',cause='1'),row('2022-01-15',dx='2')])['label_provisional']==''
    assert outcome([row('2022-01-15',dx='3',cause='1'),row('2023-01-15')])['AD_reversion_any_followup']==1
    assert not make_cohort({'DXSUM':[row('2020-01-15','m06')]},set(),cfg)[0]
    assert not make_cohort({'DXSUM':[b]}, {'synthetic'},cfg)[0]
    assert diagnosis(row('2020-01-15',dx='10',phase='TEAM'))=='UNKNOWN'
    later={'RID':'synthetic','PHASE':'ADNI2','VISCODE2':'bl','VISDATE':'2020-01-16','MMSCORE':'30','_row':3}
    assert select_measurement('MMSE',[later],b,cfg)[1] is None
    earlier=deepcopy(later);earlier['VISDATE']='2020-01-14'
    assert select_measurement('MMSE',[earlier],b,cfg)[0]['MMSCORE']==30
    conflict=deepcopy(earlier);conflict['MMSCORE']='20'
    assert select_measurement('MMSE',[earlier,conflict],b,cfg)[2]=='conflicting_latest_records'
    mri={'PHASE':'ADNI2','VISCODE2':'sc','EXAMDATE':'2020-01-14','RUNDATE':'2021-01-01','STATUS':'complete','OVERALLQC':'Pass','HIPPOQC':'Pass','_row':4}
    assert select_measurement('UCSFFSX7',[mri],b,cfg)[1] is None
    cfg['lookback_days']['NEUROBAT']=90
    lm={'PHASE':'ADNI2','VISCODE2':'sc','VISDATE':'2020-01-01','LIMMTOTAL':'12','LDELTOTAL':'8','_row':5}
    ravlt={'PHASE':'ADNI2','VISCODE2':'bl','VISDATE':'2020-01-15','AVTOT1':'5','_row':6}
    assert logical_memory([lm,ravlt],b,cfg)[0]['LDELTOTAL']==8
    print('16 focused diagnosis, time boundary, conflict, baseline and leakage checks passed.')


if __name__=='__main__':main()
