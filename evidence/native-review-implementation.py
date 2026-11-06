"""Review all matched semiconductor retaining clip attempts, without selection."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def rms(a):
    return float(np.sqrt(np.mean(np.sum(np.asarray(a)**2, axis=-1)))) if len(a) else None


def review(root):
    records = []
    initial = {}
    for completed in sorted(root.glob('*/batch_*/COMPLETED.json')):
        batch = completed.parent
        for line in (batch / 'results.jsonl').read_text().splitlines():
            row = json.loads(line)
            case = row['case_id']
            with np.load(batch / f'case_{case}.npz') as d:
                phase = d['phase_before']
                insertion = np.isin(phase, ['descend', 'cart_descend'])
                release = np.isin(phase, ['unload', 'release', 'readjust'])
                force = np.linalg.norm(d['force'], axis=1)
                pos = d['retaining_clip_pos']
                delta = d['retaining_clip_delta']
                dt = 1 / 60
                speed = np.linalg.norm(np.diff(pos, axis=0) / dt, axis=1)
                track = np.linalg.norm(d['ee_pos'] - d['ee_cmd'], axis=1) * 1000
                # Use continuous adjacent insertion samples; no artificial jumps
                # across gaps between phases enter path/impulse/acceleration.
                adjacent = insertion[1:] & insertion[:-1]
                row.update({
                    'insertion_tracking_peak_mm': float(track[insertion].max()) if insertion.any() else None,
                    'insertion_raw_force_impulse_Ns': float(np.sum((force[1:] + force[:-1])[adjacent] * .5 * dt)),
                    'insertion_retaining_clip_lateral_path_mm': float(np.linalg.norm(np.diff(pos[:, :2], axis=0)[adjacent], axis=1).sum()*1000),
                    'release_raw_force_peak_N': float(force[release].max()) if release.any() else None,
                    'terminal_1s_position_peak_mm': float(np.linalg.norm(delta[-60:], axis=1).max()*1000),
                    'terminal_1s_speed_rms_mm_s': float(np.sqrt(np.mean(speed[-59:]**2))*1000),
                    'terminal_1s_p2p_mm': float(np.linalg.norm(np.ptp(pos[-60:], axis=0))*1000),
                })
                # Physical motion is reported separately from command tracking:
                # shrinking a command error alone does not prove less shaking.
                cart = phase == 'cart_descend'
                xy = pos[cart, :2] * 1000
                if len(xy) > 22:
                    smoothed = np.column_stack([np.convolve(xy[:, j], np.ones(11)/11,
                                                            mode='same') for j in range(2)])
                    residual = (xy-smoothed)[11:-11]
                    row['retaining_clip_cart_lateral_residual_rms_mm'] = rms(residual)
                    row['retaining_clip_cart_lateral_residual_peak_mm'] = float(np.linalg.norm(residual,axis=1).max())
                else:
                    row['retaining_clip_cart_lateral_residual_rms_mm'] = None
                    row['retaining_clip_cart_lateral_residual_peak_mm'] = None
                q = row['native_quality']
                # These upstream generator defaults were previously logged but
                # not enforced. This explicitly named diagnostic is NOT the
                # pre-existing nine-metric fullcycle quality score.
                row['native_plus_declared_jitter_gate'] = bool(
                    row['native_force_quality_success'] and q['n'] >= 30
                    and q['jitter_rms'] <= .5 and q['jitter_peak'] <= 2
                    and row['duration_s'] <= 45)
                row['tracking_5mm_pass'] = bool(track.max() <= 5)
                unload = np.flatnonzero(phase == 'unload')
                withdrawal = (float(np.linalg.norm(np.asarray(row['terminal']['final_ee']) - d['ee_pos'][unload[0]]))
                              if len(unload) else None)
                row['terminal_displacement_from_unload_mm'] = withdrawal*1000 if withdrawal is not None else None
                row['terminal_100mm_displacement_gate'] = bool(withdrawal is not None and withdrawal >= .1)
                row['terminal_1s_translational_stability_diagnostic'] = bool(
                    row['terminal_1s_position_peak_mm'] <= 2 and row['terminal_1s_speed_rms_mm_s'] <= 2
                    and row['terminal_1s_p2p_mm'] <= .5)
                # This necessary gate can disprove fullcycle success. It cannot
                # establish rotation/contact/finger-clearance compliance.
                row['fullcycle_failed_necessary_gate'] = bool(not row['terminal_100mm_displacement_gate'])
            records.append(row)
            initial[(row['method'], case)] = json.loads((batch / f'case_{case}_initial.json').read_text())
    summaries = {}
    for method in sorted({r['method'] for r in records}):
        es = [r for r in records if r['method'] == method]
        summary = {'n': len(es), 'native_success': sum(r['native_force_quality_success'] for r in es),
                   'native_plus_jitter': sum(r['native_plus_declared_jitter_gate'] for r in es),
                   'tracking_5mm_pass': sum(r['tracking_5mm_pass'] for r in es),
                   'goal_fullcycle_qualified': None,
                   'failed_100mm_necessary_gate': sum(r['fullcycle_failed_necessary_gate'] for r in es),
                   'terminal_1s_translational_stability_diagnostic': sum(r['terminal_1s_translational_stability_diagnostic'] for r in es),
                   'per_type': {t: {'n': sum(r['task']==t for r in es),
                       'native_success': sum(r['native_force_quality_success'] for r in es if r['task']==t)}
                       for t in ['lab1','lab2','lab3']}}
        for key in ['duration_s', 'all_phase_force_peak_N', 'insertion_tracking_peak_mm',
                    'release_retaining_clip_xy_displacement_mm', 'final_position_error_mm',
                    'insertion_raw_force_impulse_Ns', 'insertion_retaining_clip_lateral_path_mm',
                    'retaining_clip_cart_lateral_residual_rms_mm',
                    'retaining_clip_cart_lateral_residual_peak_mm']:
            values = [r[key] for r in es if r.get(key) is not None]
            summary[key] = {'median_all_attempts': float(np.median(values)),
                            'p95_all_attempts': float(np.percentile(values,95))} if values else None
        for key in ['jitter_rms','jitter_peak','force_max','force_med_max']:
            values=[r['native_quality'][key] for r in es]
            summary[key]={'median_all_attempts':float(np.median(values)),
                          'p95_all_attempts':float(np.percentile(values,95))}
        summaries[method]=summary
    pairing=[]
    for method in summaries:
        if method=='v2': continue
        a={r['case_id']:r for r in records if r['method']==method}
        b={r['case_id']:r for r in records if r['method']=='v2'}
        matched=sorted(a.keys() & b.keys())
        maxima={key:0. for key in ['joint_positions','joint_velocities','ee_pos','retaining_clip_pos','force','torque']}
        metadata_equal=True
        for case in matched:
            ia,ib=initial[(method,case)],initial[('v2',case)]
            for key in maxima:
                maxima[key]=max(maxima[key],float(np.max(np.abs(np.array(ia['observation'][key])-ib['observation'][key]))))
            for key in ['offset','entry_offset','search_seed']:
                metadata_equal &= ia[key]==ib[key]
        wins=sum(a[c]['native_force_quality_success'] and not b[c]['native_force_quality_success'] for c in matched)
        losses=sum(b[c]['native_force_quality_success'] and not a[c]['native_force_quality_success'] for c in matched)
        pairing.append({'method':method,'reference':'v2','paired_n':len(matched),'wins':wins,'losses':losses,
            'exact_mcnemar_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
            'initial_max_abs_differences':maxima,'case_metadata_equal':bool(metadata_equal),
            'observed_initial_state_exact':bool(metadata_equal and all(x==0 for x in maxima.values()))})
    result={'scope':'Matched upstream native controller benchmark; simulation truth access in all methods.',
            'fullcycle_note':'The existing nine-metric compensated-wrench fullcycle quality score is not certified by native demo termination. Do not replace it with native_plus_declared_jitter_gate.',
            'summaries':summaries,'pairing':pairing,'episodes':records}
    (root/'review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps({'summaries':summaries,'pairing':pairing},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);args=ap.parse_args()
    review(args.root)
