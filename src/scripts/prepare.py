#!/usr/bin/env python3
"""Prepare separate four-modality overlap reports. Standard library; no training."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
from dataset_paths import project_path, recorded_hash


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'dataset/processed'
MRI_FIELDS = ['ST29SV', 'ST88SV', 'ST10CV', 'ST24CV', 'ST83CV', 'ST24TA', 'ST83TA']
CSF_FIELDS = ['ABETA42', 'TAU', 'PTAU']
PRODUCTS = {'fs7': 'UCSFFSX7', 'fs51': 'UCSFFSX51', 'csf': 'UPENNBIOMK_ROCHE_ELECSYS'}
INITIAL_VISITS = {'sc', 'scmri', 'bl'}
MISSING = {'', 'NA', 'N/A', 'NaN', 'nan', 'NULL', '-1', '-4', '-1.0', '-4.0'}
# The inclusive retrospective policy isolates the same-day boundary effect.
POLICIES = {
    'strict': {'last_collection_day': 0, 'processing_by_baseline': True},
    'retrospective': {'last_collection_day': -1, 'processing_by_baseline': False},
    'retrospective_same_day': {'last_collection_day': 0, 'processing_by_baseline': False},
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path, required=()):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if not set(required) <= set(headers):
            raise ValueError(f'Missing columns in {path.name}: {set(required) - set(headers)}')
        rows = []
        for number, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'Malformed CSV row {number} in {path.name}')
            rows.append({key: value.strip() for key, value in row.items()})
        return headers, rows


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def write_rows(path, rows):
    if not rows:
        raise ValueError(f'Explicit headers required for empty output: {path.name}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(path, rows, fields)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def parse_date(value):
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def number(value):
    if value is None or value in MISSING:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def observed(value):
    return value is not None and value != ''


def indexed(rows, name):
    result = {row['RID']: row for row in rows}
    if '' in result or len(result) != len(rows):
        raise ValueError(f'Missing or duplicate RID in {name}')
    return result


def clean_values(kind, row, cfg):
    fields = CSF_FIELDS if kind == 'csf' else MRI_FIELDS
    values, flags = {}, []
    for field in fields:
        value = number(row[field])
        if value is None:
            flags.append(field + ':missing_or_invalid')
        elif kind != 'csf' and value <= 0:
            value = None
            flags.append(field + ':nonpositive')
        elif kind == 'csf' and not cfg['csf_limits'][field][0] <= value <= cfg['csf_limits'][field][1]:
            value = None
            flags.append(field + ':outside_technical_range')
        values[field] = value
    if kind == 'csf':
        comment = row.get('COMMENT', '').lower()
        if row['BATCH'] not in cfg['csf_verified_batches']:
            values = dict.fromkeys(fields)
            flags.append('unverified_batch')
        if 'hemoly' in comment or 'failed' in comment:
            values = dict.fromkeys(fields)
            flags.append('specimen_or_recalculation_problem')
        for match in re.finditer(r'(abeta42|ptau|tau)\s*[<>≤≥]', comment):
            field = match.group(1).upper()
            values[field] = None
            flags.append(field + ':censored_comment')
    if row.get('HAS_QC_ERROR', '') not in {'', '0'}:
        values = dict.fromkeys(fields)
        flags.append('CRF_QC_error')
    return values, flags


def quality_ok(kind, row):
    if row.get('HAS_QC_ERROR', '') not in {'', '0'}:
        return False
    if kind == 'csf':
        return True
    if row['OVERALLQC'].lower() != 'pass':
        return False
    if kind == 'fs7':
        return row['STATUS'].lower() == 'complete' and row['HIPPOQC'].lower() == 'pass'
    # 5.1's LHIPQC/RHIPQC describe unselected subfields. Overall Pass governs
    # this whole-region bundle. STATUS is audited, not given the 7.x meaning.
    return True


def load_frozen(cfg):
    folder = project_path(cfg['frozen_dir'])
    run = json.loads(project_path(cfg['frozen_model_manifest']).read_text())
    if run.get('status') != 'complete':
        raise ValueError('A completed original modeling manifest is required')
    paths = [folder / name for name in ('baseline_predictors.csv', 'outcome_audit.csv', 'predictor_schema.json')]
    paths.append(project_path(cfg['preparation_manifest']))
    for path in paths:
        key = str(path.relative_to(ROOT))
        if recorded_hash(run['input_hashes'], path) != sha256(path):
            raise ValueError('Frozen input differs from the original model run: ' + key)
    prep = json.loads(paths[-1].read_text())['config']
    if (prep['horizon_months'] != 36 or prep['followup_tolerance_days'] != 0
            or prep['apoe_policy'] != 'tested_by_baseline'
            or prep['availability_policy'] != 'result_by_baseline'
            or any(days != 90 for days in prep['lookback_days'].values())):
        raise ValueError('Expected the saved strict 36-month, 90-day preparation')
    schema = json.loads((folder / 'predictor_schema.json').read_text())
    clinical, apoe = schema['modalities']['clinical'], schema['modalities']['APOE']
    if len(clinical) != 17 or len(apoe) != 1 or set(clinical) & set(apoe):
        raise ValueError('Unexpected frozen clinical/APOE schema')
    features = clinical + apoe
    _, predictors = read_csv(folder / 'baseline_predictors.csv', ['RID', 'PTID'] + features)
    _, outcomes = read_csv(folder / 'outcome_audit.csv', [
        'RID', 'PTID', 'baseline_date', 'baseline_phase', 'horizon_months',
        'label_provisional', 'outcome_reason'])
    bypred, byout = indexed(predictors, 'predictors'), indexed(outcomes, 'outcomes')
    if set(bypred) != set(byout) or len({row['PTID'] for row in predictors}) != len(predictors):
        raise ValueError('Frozen participant identities disagree')
    for rid, outcome in byout.items():
        if (outcome['label_provisional'] not in {'', '0', '1'}
                or outcome['horizon_months'] != '36' or not parse_date(outcome['baseline_date'])
                or not outcome['PTID'] or outcome['PTID'] != bypred[rid]['PTID']):
            raise ValueError('Invalid frozen outcome/identity record')
        for feature in features:
            value = bypred[rid][feature]
            if observed(value) and number(value) is None:
                raise ValueError('Invalid frozen predictor: ' + feature)
    return bypred, byout, {'clinical': clinical, 'APOE': apoe}, paths


def load_sources(cfg, outcomes):
    data, inventory = {}, []
    for kind, config_key in [('fs7', 'mri_fs7'), ('fs51', 'mri_fs51'), ('csf', 'csf')]:
        path = project_path(cfg[config_key])
        required = ['RID', 'VISCODE2', 'EXAMDATE', 'RUNDATE']
        required += CSF_FIELDS + ['PHASE', 'PTID', 'BATCH', 'COMMENT'] if kind == 'csf' else MRI_FIELDS + ['IMAGEUID', 'STATUS', 'OVERALLQC']
        required += ['COLPROT', 'LHIPQC', 'RHIPQC'] if kind == 'fs51' else []
        required += ['PHASE', 'PTID', 'HIPPOQC'] if kind == 'fs7' else []
        header, rows = read_csv(path, required)
        for line, row in enumerate(rows, 2):
            row['_row'] = line
            row['_file'] = str(path.relative_to(ROOT))
        data[kind] = rows
        ids = {row['RID'] for row in rows if row['RID']}
        master = ids & set(outcomes)
        inventory.append({'source': kind, 'file': str(path.relative_to(ROOT)), 'rows': len(rows),
                          'columns': len(header), 'unique_RIDs': len(ids), 'master_RIDs': len(master),
                          'resolved_label_RIDs': sum(outcomes[rid]['label_provisional'] != '' for rid in master),
                          'sha256': sha256(path)})
    _, dictionary = read_csv(project_path(cfg['dictionary']), ['TBLNAME', 'FLDNAME'])
    dictionary_keys = {(row['TBLNAME'], row['FLDNAME']) for row in dictionary}
    for kind, table in PRODUCTS.items():
        for field in CSF_FIELDS if kind == 'csf' else MRI_FIELDS:
            if (table, field) not in dictionary_keys:
                raise ValueError(f'Missing dictionary definition: {table}.{field}')
    return data, inventory


def source_facts(data, outcomes, crosswalk, image_rows, cfg):
    images = defaultdict(list)
    for row in image_rows:
        images[row['image_id']].append(row)
    rid_to_ptid, ptid_to_rid = defaultdict(set), defaultdict(set)
    for row in list(outcomes.values()) + data['fs7'] + data['csf']:
        if row.get('RID') and row.get('PTID'):
            rid_to_ptid[row['RID']].add(row['PTID'])
            ptid_to_rid[row['PTID']].add(row['RID'])
    bad = {rid for rid, ids in rid_to_ptid.items() if len(ids) != 1}
    for ids in ptid_to_rid.values():
        if len(ids) != 1:
            bad.update(ids)
    result, audit = {kind: defaultdict(list) for kind in data}, []
    for kind, rows in data.items():
        for row in rows:
            rid = row['RID']
            if rid not in outcomes:
                continue
            outcome = outcomes[rid]
            baseline = parse_date(outcome['baseline_date'])
            collected, processed = parse_date(row['EXAMDATE']), parse_date(row['RUNDATE'])
            offset = (collected - baseline).days if collected else None
            link = 'not_applicable'
            if kind != 'csf':
                matches = images.get(row['IMAGEUID'].removeprefix('I'), [])
                link = ('missing' if not matches else 'multiple' if len(matches) > 1 else
                        'matched' if matches[0]['subject_id'] == outcome['PTID'] and
                        matches[0]['image_date'] == row['EXAMDATE'] else 'participant_or_date_conflict')
            id_ok = rid not in bad and crosswalk.get(rid, {}).get('consistent') == '1'
            id_ok = id_ok and crosswalk.get(rid, {}).get('PTID') == outcome['PTID']
            if kind != 'fs51':
                id_ok = id_ok and row['PTID'] == outcome['PTID']
            # Missing legacy Key MRI matches are audit-only; a positive contradiction is not.
            id_ok = id_ok and link != 'participant_or_date_conflict'
            phase = row['COLPROT'] if kind == 'fs51' else row['PHASE']
            values, flags = clean_values(kind, row, cfg)
            facts = {
                'identifier_consistent': id_ok,
                'same_phase_initial_visit': phase == outcome['baseline_phase'] and row['VISCODE2'] in INITIAL_VISITS,
                'known_collection_date': collected is not None,
                'known_processing_date': processed is not None,
                'processing_not_before_collection': collected is not None and processed is not None and processed >= collected,
                'processing_by_baseline': processed is not None and processed <= baseline,
                'QC_pass': quality_ok(kind, row),
                'verified_batch': kind != 'csf' or row['BATCH'] in cfg['csf_verified_batches'],
            }
            record = {'row': row, 'offset': offset, 'values': values, 'flags': flags, 'facts': facts, 'link': link}
            result[kind][rid].append(record)
            audit.append({'source': kind, 'RID': rid, 'source_file': row['_file'], 'source_row': row['_row'],
                          'source_phase': phase, 'VISCODE2': row['VISCODE2'], 'collection_date': row['EXAMDATE'],
                          'processing_date': row['RUNDATE'], 'collection_offset_days': offset,
                          'processing_offset_days': (processed - baseline).days if processed else None,
                          'key_mri_link': link, 'STATUS': row.get('STATUS', ''), 'OVERALLQC': row.get('OVERALLQC', ''),
                          'HIPPOQC': row.get('HIPPOQC', ''), 'LHIPQC': row.get('LHIPQC', ''), 'RHIPQC': row.get('RHIPQC', ''),
                          'BATCH': row.get('BATCH', ''), 'COMMENT': row.get('COMMENT', ''),
                          'cleaning_flags': ';'.join(flags), **{key: int(value) for key, value in facts.items()}})
    return result, audit


def time_gates(policy, lookback):
    spec = POLICIES[policy]
    gates = [(key, lambda record, key=key: record['facts'][key]) for key in [
        'identifier_consistent', 'same_phase_initial_visit', 'known_collection_date']]
    gates.append(('collection_window', lambda record: record['offset'] is not None and
                  -lookback <= record['offset'] <= spec['last_collection_day']))
    gates += [(key, lambda record, key=key: record['facts'][key]) for key in [
        'known_processing_date', 'processing_not_before_collection']]
    if spec['processing_by_baseline']:
        gates.append(('processing_by_baseline', lambda record: record['facts']['processing_by_baseline']))
    return gates


def choose(kind, records, policy, lookback):
    candidates = records
    for _, predicate in time_gates(policy, lookback):
        candidates = [record for record in candidates if predicate(record)]
    # MRI chooses among eligible QC-passed scans. CSF chooses the first known
    # assay at the selected timepoint BEFORE value/QC cleaning or batch filtering.
    if kind != 'csf':
        candidates = [record for record in candidates if record['facts']['QC_pass']]
    if not candidates:
        return None, 'no_eligible_record', 0
    latest = max(record['row']['EXAMDATE'] for record in candidates)
    tied = [record for record in candidates if record['row']['EXAMDATE'] == latest]
    if kind == 'csf':
        earliest = min(record['row']['RUNDATE'] for record in tied)
        tied = [record for record in tied if record['row']['RUNDATE'] == earliest]
        compare = CSF_FIELDS + ['BATCH', 'COMMENT', 'HAS_QC_ERROR']
    else:
        compare = MRI_FIELDS + ['IMAGEUID']
    if len({tuple(record['row'].get(field, '') for field in compare) for record in tied}) != 1:
        return None, 'conflicting_selected_records', len(candidates)
    chosen = min(tied, key=lambda record: (record['row']['RUNDATE'], record['row']['_row']))
    return chosen, 'selected', len(candidates)


def selected_measurements(indexes, outcomes, cfg):
    selected, provenance, attrition = {}, [], []
    for policy in POLICIES:
        for kind, byrid in indexes.items():
            key = (policy, kind)
            selected[key] = {}
            pool = [(rid, record) for rid, records in byrid.items() for record in records]
            def count_stage(stage, rows):
                ids = {rid for rid, _ in rows}
                attrition.append({'policy': policy, 'source': kind, 'stage': stage, 'source_rows': len(rows),
                                  'participants': len(ids), 'resolved_outcomes': sum(outcomes[rid]['label_provisional'] != '' for rid in ids)})
            count_stage('source_records_in_master', pool)
            for stage, predicate in time_gates(policy, cfg['lookback_days']):
                pool = [(rid, record) for rid, record in pool if predicate(record)]
                count_stage(stage, pool)
            if kind != 'csf':
                pool = [(rid, record) for rid, record in pool if record['facts']['QC_pass']]
                count_stage('QC_pass', pool)
            chosen_pool = []
            for rid, outcome in outcomes.items():
                record, reason, count = choose(kind, byrid.get(rid, []), policy, cfg['lookback_days'])
                values = dict.fromkeys(CSF_FIELDS if kind == 'csf' else MRI_FIELDS)
                if record:
                    values = record['values']
                    chosen_pool.append((rid, record))
                selected[key][rid] = values
                row = record['row'] if record else {}
                provenance.append({'policy': policy, 'source': kind, 'RID': rid, 'reason': reason,
                                   'candidate_count': count, 'source_file': row.get('_file', ''),
                                   'source_row': row.get('_row', ''), 'collection_date': row.get('EXAMDATE', ''),
                                   'processing_date': row.get('RUNDATE', ''), 'collection_offset_days': record['offset'] if record else '',
                                   'BATCH': row.get('BATCH', ''), 'IMAGEUID': row.get('IMAGEUID', ''),
                                   'STATUS': row.get('STATUS', ''), 'OVERALLQC': row.get('OVERALLQC', ''),
                                   'key_mri_link': record['link'] if record else '',
                                   'cleaning_flags': ';'.join(record['flags']) if record else '',
                                   'available_any': int(any(observed(value) for value in values.values())),
                                   'complete': int(all(observed(value) for value in values.values()))})
            count_stage('selected_record', chosen_pool)
            pool = [(rid, record) for rid, record in chosen_pool if record['facts']['QC_pass'] and record['facts']['verified_batch']]
            count_stage('selected_record_QC_and_verified_batch', pool)
            pool = [(rid, record) for rid, record in pool if any(observed(value) for value in record['values'].values())]
            count_stage('selected_record_any_valid_value', pool)
            pool = [(rid, record) for rid, record in pool if all(observed(value) for value in record['values'].values())]
            count_stage('selected_record_complete', pool)
    return selected, provenance, attrition


def make_datasets(out, frozen, outcomes, base_modalities, selected):
    coverage, eligibility, phases, four = [], [], [], []
    for policy in POLICIES:
        for mri in ['fs7', 'fs51']:
            scenario = policy + '/' + mri
            folder = out / 'datasets' / policy / mri
            folder.mkdir(parents=True)
            modalities = {'clinical': base_modalities['clinical'], 'APOE': base_modalities['APOE'],
                          'MRI': [PRODUCTS[mri] + '__' + field for field in MRI_FIELDS],
                          'CSF': [PRODUCTS['csf'] + '__' + field for field in CSF_FIELDS]}
            features = [feature for fields in modalities.values() for feature in fields]
            master, flags, complete_cases, exports = [], {}, [], []
            for rid, previous in frozen.items():
                row = {field: previous[field] for field in ['RID', 'PTID'] + modalities['clinical'] + modalities['APOE']}
                for kind, modality in [(mri, 'MRI'), ('csf', 'CSF')]:
                    row.update({PRODUCTS[kind] + '__' + field: value for field, value in selected[(policy, kind)][rid].items()})
                master.append(row)
                availability = {definition: {modality: (any if definition == 'any' else all)(observed(row[field]) for field in fields)
                                               for modality, fields in modalities.items()} for definition in ['any', 'complete']}
                flags[rid] = availability
                label = outcomes[rid]['label_provisional']
                eligibility.append({'scenario': scenario, 'RID': rid, 'label_available': int(label != ''),
                                    'label_provisional': label, 'outcome_reason': outcomes[rid]['outcome_reason'],
                                    **{definition + '_' + modality: int(value) for definition, values in availability.items() for modality, value in values.items()},
                                    'four_modalities_complete_labeled': int(label != '' and all(availability['complete'].values()))})
                if label != '' and all(availability['complete'].values()):
                    complete_cases.append({**row, 'label': int(label)})
            master_by_rid = {row['RID']: row for row in master}
            for size in range(1, 5):
                for combo in itertools.combinations(modalities, size):
                    for definition in ['any', 'complete']:
                        available = [rid for rid in frozen if all(flags[rid][definition][modality] for modality in combo)]
                        known = [rid for rid in available if outcomes[rid]['label_provisional'] != '']
                        positives = sum(outcomes[rid]['label_provisional'] == '1' for rid in known)
                        item = {'scenario': scenario, 'policy': policy, 'mri_product': PRODUCTS[mri],
                                'modalities': '+'.join(combo), 'definition': definition, 'master_available': len(available),
                                'labeled': len(known), 'converters': positives, 'nonconverters': len(known) - positives,
                                'unresolved_outcome': len(available) - len(known)}
                        coverage.append(item)
                        if definition == 'complete':
                            selected_features = [feature for modality in combo for feature in modalities[modality]]
                            filename = 'complete_cases/' + '_'.join(combo) + '.csv'
                            export_rows = [
                                {**{field: master_by_rid[rid][field] for field in ['RID', 'PTID'] + selected_features},
                                 'label': int(outcomes[rid]['label_provisional'])}
                                for rid in known
                            ]
                            write_csv(folder / filename, export_rows, ['RID', 'PTID'] + selected_features + ['label'])
                            exports.append({'file': filename, 'modalities': list(combo),
                                            'predictor_columns': selected_features, 'participants': len(known),
                                            'converters': positives, 'nonconverters': len(known) - positives})
                        if size == 4 and definition == 'complete':
                            four.append({**item, 'class_support': 'empty' if not known else 'single_class' if positives in {0, len(known)} else 'both_classes_sample_size_review_needed'})
                        for phase in sorted({outcomes[rid]['baseline_phase'] for rid in available}):
                            ids = [rid for rid in available if outcomes[rid]['baseline_phase'] == phase]
                            phases.append({'scenario': scenario, 'modalities': '+'.join(combo), 'definition': definition,
                                           'baseline_phase': phase, 'master_available': len(ids),
                                           'converters': sum(outcomes[rid]['label_provisional'] == '1' for rid in ids),
                                           'nonconverters': sum(outcomes[rid]['label_provisional'] == '0' for rid in ids),
                                           'unresolved_outcome': sum(outcomes[rid]['label_provisional'] == '' for rid in ids)})
            write_csv(folder / 'baseline_predictors.csv', master, ['RID', 'PTID'] + features)
            write_csv(folder / 'four_modality_complete_cases.csv', complete_cases, ['RID', 'PTID'] + features + ['label'])
            write_json(folder / 'predictor_schema.json', {'scenario': scenario, 'mri_product': PRODUCTS[mri],
                       'modalities': modalities, 'identifiers': ['RID', 'PTID'], 'predictor_columns': features,
                       'target': 'label', 'excluded_from_X': 'All columns not explicitly listed in predictor_columns.',
                       'outcome_file': '../../../outcome_audit.csv', 'policies': POLICIES[policy],
                       'complete_case_exports': exports})
    return coverage, eligibility, phases, four


def report_text(outcomes, coverage, four):
    lines = ['# Four-modality overlap results', '',
             'Generated from frozen clinical/APOE predictors and outcomes plus the new MRI and core CSF sources. No model training.', '',
             f"Master: {len(outcomes)} participants. Resolved outcomes: {sum(row['label_provisional'] != '' for row in outcomes.values())}.", '',
             '## All four modalities: complete predictors and resolved outcomes', '',
             '| Scenario | Complete in master | Labeled | Converters | Non-converters | Unresolved | Class support |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in four:
        lines.append('| ' + ' | '.join(str(row[key]) for key in ['scenario', 'master_available', 'labeled', 'converters', 'nonconverters', 'unresolved_outcome', 'class_support']) + ' |')
    lines += ['', 'Both classes being present does not establish sufficient statistical power or suitable validation folds.', '',
              '## Complete-case labeled availability for every combination', '',
              '| Modalities | ' + ' | '.join(row['scenario'] for row in four) + ' |',
              '|---|' + '---:|' * len(four)]
    combos = list(dict.fromkeys(row['modalities'] for row in coverage))
    lookup = {(row['scenario'], row['modalities']): row['labeled'] for row in coverage if row['definition'] == 'complete'}
    for combo in combos:
        lines.append('| ' + combo + ' | ' + ' | '.join(str(lookup[(row['scenario'], combo)]) for row in four) + ' |')
    lines += ['', '## Interpretation', '',
              '- Strict: collection in [-90, 0] days and processing by baseline. This uses refreshed MRI and the new core CSF definition, so it does not reproduce the old batch-12-only dataset.',
              '- Retrospective: collection in [-90, -1] days; later processing is allowed. It does not establish result availability at baseline.',
              '- Retrospective_same_day: collection in [-90, 0] days with later processing allowed. Date-only records do not establish within-day ordering; this is a boundary sensitivity.',
              '- Clinical and APOE rules remain the frozen original rules in every scenario, including inclusive same-day clinical matching and tested-by-baseline APOE.',
              '- MRI versions are separate; no version substitution or pooling was performed.',
              '- FS7 requires complete status, overall Pass and HIPPOQC Pass. FS51 requires overall Pass for the seven whole-region variables; its STATUS and subfield QC are audited separately.',
              '- CSF consists of Aβ42, total tau and p-tau181. Batch 9 does not require its unmeasured Aβ40.',
              '- Unknown outcomes remain missing. All modeling tables require a resolved label and complete requested predictors.',
              '- Coverage is descriptive. More available records do not establish higher model accuracy.', '',
              '## Files', '',
              '- `four_modality_summary.csv`: the main six-row result.',
              '- `overlap.csv`: all 15 nonempty modality combinations, any/complete definitions, and six scenarios.',
              '- `participant_availability.csv`: participant-level flags, kept separate from predictors.',
              '- `cohort_by_phase.csv`: class counts and unresolved outcomes by phase.',
              '- `attrition.csv`: sequential, order-dependent losses; CSF chooses a collection/first assay before cleaning.',
              '- `source_record_audit.csv`, `measurement_selection.csv`, `selection_summary.csv`: dates, QC, invalid values and selected source provenance.',
              '- `input_inventory.csv`: whole-source counts; these are not eligible sample sizes.',
              '- `../datasets/<policy>/<fs7-or-fs51>/`: full master predictors, schema and labeled complete four-modality cases.',
              '- `../datasets/<policy>/<fs7-or-fs51>/complete_cases/`: labeled complete-case CSVs for all 15 modality combinations. Each contains only its requested predictors, identifiers and label; feature lists and counts are in the scenario schema.',
              '- `../outcome_audit.csv`: byte-identical frozen outcomes.',
              '- `../clinical_apoe_provenance.csv`: original provenance for the reused predictors.',
              '- `../manifest.json`: configuration, source/code fingerprints and completion status.', '']
    return '\n'.join(lines)


def main(config_path, output_override=None):
    cfg = json.loads(config_path.read_text())
    if cfg['lookback_days'] != 90 or set(cfg['csf_verified_batches']) != {'UPENNBIOMK9', 'UPENNBIOMK11', 'UPENNBIOMK12', 'UPENNBIOMK13'}:
        raise ValueError('This workflow implements the documented 90-day, four-batch core CSF review')
    if cfg['csf_limits'] != {'ABETA42': [200, 1700], 'TAU': [80, 1300], 'PTAU': [8, 120]}:
        raise ValueError('Core assay limits differ from the reviewed methods')
    out = project_path(output_override or cfg['output_dir'])
    if out == WORK or not out.is_relative_to(WORK.resolve()):
        raise ValueError('Outputs must be in a child directory of dataset/processed')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Output already exists; use --output dataset/processed/multimodal-v2 for a new run')
    frozen, outcomes, modalities, frozen_paths = load_frozen(cfg)
    protected = set(frozen_paths + [config_path, Path(__file__).resolve()])
    protected.update(project_path(cfg[key]) for key in ['frozen_model_manifest', 'identifier_crosswalk', 'mri_fs7', 'mri_fs51', 'csf', 'key_mri', 'dictionary', 'csf_dictionary'])
    protected.update(project_path(value) for value in cfg['methods'])
    provenance_path = project_path(cfg['frozen_dir']) / 'measurement_provenance.csv'
    protected.add(provenance_path)
    if any(path == out or path.is_relative_to(out) for path in protected):
        raise ValueError('Output directory contains protected inputs')
    before = {str(path.relative_to(ROOT)): sha256(path) for path in sorted(protected)}
    data, inventory = load_sources(cfg, outcomes)
    _, crosswalk = read_csv(project_path(cfg['identifier_crosswalk']), ['RID', 'PTID', 'consistent'])
    _, images = read_csv(project_path(cfg['key_mri']), ['image_id', 'subject_id', 'image_date'])
    indexes, audit = source_facts(data, outcomes, indexed(crosswalk, 'crosswalk'), images, cfg)
    selected, selections, attrition = selected_measurements(indexes, outcomes, cfg)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'status': 'running', 'config': cfg, 'actual_output_dir': str(out.relative_to(ROOT)),
                'policies': POLICIES, 'python': sys.version, 'input_sha256': before,
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'master_participants': len(outcomes), 'frozen_clinical_APOE': True, 'frozen_outcomes': True,
                'fs51_QC': 'Overall Pass; STATUS and hippocampal subfield QC are audit-only for the selected whole-region bundle.',
                'CSF_selection': 'Latest eligible collection date, then earliest known assay; tied disagreement withheld; no completeness-based fallback.'}
    write_json(out / 'manifest.json', manifest)
    try:
        reports = out / 'reports'
        reports.mkdir()
        coverage, eligibility, phases, four = make_datasets(out, frozen, outcomes, modalities, selected)
        for name, rows in [('overlap', coverage), ('participant_availability', eligibility), ('four_modality_summary', four),
                           ('attrition', attrition), ('input_inventory', inventory), ('measurement_selection', selections)]:
            write_rows(reports / (name + '.csv'), rows)
        write_csv(reports / 'cohort_by_phase.csv', phases, ['scenario', 'modalities', 'definition', 'baseline_phase', 'master_available', 'converters', 'nonconverters', 'unresolved_outcome'])
        if audit:
            write_rows(reports / 'source_record_audit.csv', audit)
        else:
            write_csv(reports / 'source_record_audit.csv', [], ['source', 'RID', 'source_row'])
        summary = Counter((row['policy'], row['source'], row['reason'], row['STATUS'], row['OVERALLQC'], row['available_any'], row['complete']) for row in selections)
        write_rows(reports / 'selection_summary.csv', [dict(zip(['policy', 'source', 'reason', 'STATUS', 'OVERALLQC', 'available_any', 'complete'], key), participants=count) for key, count in sorted(summary.items())])
        outcome_path = project_path(cfg['frozen_dir']) / 'outcome_audit.csv'
        (out / 'outcome_audit.csv').write_bytes(outcome_path.read_bytes())
        headers, provenance = read_csv(provenance_path, ['RID', 'table'])
        keep = {'PTDEMOG', 'MMSE', 'ADAS', 'NEUROBAT', 'NEUROBAT_LOGICAL_MEMORY', 'FAQ', 'APOERES'}
        write_csv(out / 'clinical_apoe_provenance.csv', [row for row in provenance if row['table'] in keep], headers)
        if any(sha256(project_path(path)) != expected for path, expected in before.items()):
            raise ValueError('An input changed during preparation')
        if sha256(out / 'outcome_audit.csv') != sha256(outcome_path):
            raise ValueError('Copied outcomes do not match frozen outcomes')
        (reports / 'README.md').write_text(report_text(outcomes, coverage, four), encoding='utf-8')
        manifest.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                        overlap_rows=len(coverage), four_modality_summary=four,
                        output_sha256={str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name != 'manifest.json'})
        write_json(out / 'manifest.json', manifest)
    except Exception as error:
        manifest.update(status='failed', error=str(error))
        write_json(out / 'manifest.json', manifest)
        raise
    print('Finished. Read ' + str((out / 'reports/README.md').relative_to(ROOT)))
    for row in four:
        print(f"{row['scenario']}: complete labeled N={row['labeled']}, converters={row['converters']}, non-converters={row['nonconverters']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/prepare_multimodal.json', help='Project-relative configuration')
    parser.add_argument('--output', help='New project-relative output directory beneath dataset/processed')
    args = parser.parse_args()
    main(project_path(args.config), args.output)
