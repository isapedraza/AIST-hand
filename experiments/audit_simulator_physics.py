"""Audit a frozen Shadow trace without editing its model or live configuration.

Run with the MuJoCo environment used to record the trace. Outputs distinguish
saved inputs, forces reconstructed with mj_forward, and counterfactual holds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return int(value)


def fields(obj, names):
    return {name: plain(getattr(obj, name)) for name in names}


def summary(values):
    a = np.asarray(values)
    return ({'count': int(a.size), 'median': float(np.median(a)),
             'p95': float(np.percentile(a, 95)), 'max': float(a.max())}
            if a.size else {'count': 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True,
                        help='Directory containing sim/model.mjb and sim/metadata.json')
    parser.add_argument('--simulator-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-stride', type=int, default=100)
    parser.add_argument('--hold-seconds', type=float, default=1.0)
    args = parser.parse_args()
    if args.sample_stride < 1 or args.hold_seconds <= 0:
        parser.error('sample-stride and hold-seconds must be positive')
    sys.path.insert(0, str(args.simulator_root.resolve()))
    sys.path.insert(0, str(args.simulator_root.resolve() / 'dexjoco'))
    from shadow_ext.state_recording import TraceReader
    from shadow_ext.build import build_spec
    from dexjoco.sim.controllers.opspace import opspace

    reader = TraceReader(args.session / 'sim')
    model = reader.model
    food = model.body('boxed_food_0').id
    bucket = model.body('bucket').id
    handle = model.body('bucket_link_0').id
    palm = model.body('rh-rh_palm').id
    hand_bodies = {i for i in range(model.nbody)
                   if model.body(i).name.startswith('rh-')}
    hand_geoms = [i for i in range(model.ngeom)
                  if model.geom_bodyid[i] in hand_bodies]
    arm_act = np.array([model.actuator(f'actuator{i}').id for i in range(1, 8)])
    arm_dof = np.array([model.joint(f'joint{i}').dofadr[0] for i in range(1, 8)])
    hand_act = np.array([i for i in range(model.nu)
                         if model.actuator(i).name.startswith('rh-')])
    site = model.site('attachment_site').id
    mocap = int(model.body('target').mocapid[0])
    home = np.array([0., -.785, 0., -2.35, 0., 1.57, np.pi / 4])
    options = {key: plain(getattr(model.opt, key)) for key in dir(model.opt)
               if not key.startswith('_') and not callable(getattr(model.opt, key))}
    arrays = ('body_mass body_inertia body_ipos body_iquat body_gravcomp body_mocapid '
              'body_parentid jnt_type jnt_bodyid jnt_range jnt_limited '
              'jnt_stiffness jnt_solref jnt_solimp dof_damping dof_armature '
              'dof_frictionloss geom_bodyid geom_type geom_size geom_pos geom_quat '
              'geom_contype geom_conaffinity geom_condim geom_friction geom_solref '
              'geom_solimp geom_margin geom_gap geom_priority geom_solmix '
              'actuator_trntype actuator_trnid actuator_gear actuator_gainprm '
              'actuator_biasprm actuator_dyntype actuator_gaintype actuator_biastype '
              'actuator_ctrllimited actuator_ctrlrange actuator_forcelimited '
              'actuator_forcerange tendon_range tendon_limited tendon_stiffness '
              'tendon_damping tendon_frictionloss tendon_adr tendon_num '
              'wrap_type wrap_objid wrap_prm eq_type eq_obj1id eq_obj2id '
              'eq_data exclude_signature pair_geom1 pair_geom2 pair_friction').split()
    out = {'schema_version': 1, 'session': str(args.session),
           'mujoco_version': mujoco.__version__,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'recording_metadata': reader.metadata,
           'options': options,
           'names': {kind: [getattr(model, kind)(i).name for i in range(count)]
                     for kind, count in [('body', model.nbody), ('geom', model.ngeom),
                                         ('joint', model.njnt), ('actuator', model.nu),
                                         ('tendon', model.ntendon)]},
           'compiled_arrays': fields(model, arrays)}
    out['controller'] = {'position_gains': [400, 400, 400],
                         'orientation_gains': [200, 200, 200], 'damping_ratio': 1,
                         'nullspace_stiffness': .5, 'gravity_compensation': True,
                         'home': home.tolist(), 'acceleration_caps': None,
                         'opspace_sha256': hashlib.sha256(
                             (args.simulator_root / 'dexjoco/dexjoco/sim/controllers/opspace.py')
                             .read_bytes()).hexdigest()}

    hand_mass = float(sum(model.body_mass[i] for i in hand_bodies))
    bucket_mass = float(model.body_mass[bucket] + model.body_mass[handle])
    out['supported_masses'] = {'shadow_subtree_kg': hand_mass,
        'shadow_forearm_kg': float(model.body('rh-rh_forearm').mass[0]),
        'bucket_with_handle_kg': bucket_mass,
        'shadow_plus_food_kg': hand_mass + float(model.body_mass[food]),
        'shadow_plus_loaded_bucket_kg': hand_mass + bucket_mass + float(model.body_mass[food])}

    # Bit order and individual mj_stateSize calls follow MuJoCo's state API.
    slices, offset = {}, 0
    for bit in range(14):
        flag = mujoco.mjtState(1 << bit)
        if int(reader.spec) & int(flag):
            size = mujoco.mj_stateSize(model, flag)
            slices[flag.name] = slice(offset, offset + size)
            offset += size
    assert offset == mujoco.mj_stateSize(model, reader.spec)
    maxima = {key: 0. for key in ('qfrc_applied', 'xfrc_applied', 'slip_bias',
                                  'object_gravcomp', 'bucket_motor_ctrl')}
    nrows = 0
    previous = None
    residuals = {'food': 0., 'bucket': 0.}
    time_step_error = 0.
    for path in reader.paths:
        with np.load(path, allow_pickle=False) as chunk:
            states = chunk['state']
            nrows += len(states)
            for key in ('qfrc_applied', 'xfrc_applied'):
                maxima[key] = max(maxima[key], float(np.max(np.abs(
                    states[:, slices['mjSTATE_' + key.upper()]]))))
            maxima['slip_bias'] = max(maxima['slip_bias'],
                                     float(np.max(np.abs(chunk['slip_bias']))))
            maxima['object_gravcomp'] = max(maxima['object_gravcomp'],
                float(np.max(np.abs(chunk['body_gravcomp'][:, [food, bucket, handle]]))))
            maxima['bucket_motor_ctrl'] = max(maxima['bucket_motor_ctrl'], float(np.max(
                np.abs(states[:, slices['mjSTATE_CTRL']][:, model.actuator('bucket_joint_0').id]))))
            consecutive = states if previous is None else np.vstack((previous, states))
            times = consecutive[:, slices['mjSTATE_TIME']].ravel()
            delta_t = np.diff(times)
            time_step_error = max(time_step_error, float(np.max(np.abs(delta_t - model.opt.timestep))))
            qpos = consecutive[:, slices['mjSTATE_QPOS']]
            qvel = consecutive[:, slices['mjSTATE_QVEL']]
            for label, name in [('food', 'boxed_food_0_freejoint'), ('bucket', 'bucket_root')]:
                j = model.joint(name)
                qp, qv = int(j.qposadr[0]), int(j.dofadr[0])
                error = np.diff(qpos[:, qp:qp + 3], axis=0) - delta_t[:, None] * qvel[1:, qv:qv + 3]
                residuals[label] = max(residuals[label], float(np.max(np.abs(error))))
            previous = states[-1:].copy()
    out['all_saved_states'] = {'count': nrows, 'max_absolute': maxima,
                               'max_timestep_error_s': time_step_error,
                               'max_free_translation_euler_residual_m': residuals,
                               'state_slices': {k: [v.start, v.stop] for k, v in slices.items()}}
    print('Verified all saved inputs', out['all_saved_states']['max_absolute'], flush=True)

    # Reconstructed contact forces are estimates at saved configurations, not
    # contact-force telemetry recorded during the original integration step.
    samples, contact_rows = [], []
    initial = reader.restore(0)
    z0 = {b: float(initial.xpos[b, 2]) for b in (food, bucket)}
    for index in sorted(set(range(0, nrows, args.sample_stride)) | {nrows - 1}):
        data = reader.restore(index)
        counts = {food: 0, bucket: 0}
        for ci, c in enumerate(data.contact):
            b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
            target = next((b for b in (food, bucket, handle) if b in (b1, b2)), None)
            if target is None or not ({b1, b2} & hand_bodies):
                continue
            counts[food if target == food else bucket] += 1
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, force)
            contact_rows.append({'index': index, 'time_s': float(data.time),
                'bodies': [model.body(b1).name, model.body(b2).name],
                'geoms': [int(c.geom1), int(c.geom2)], 'dim': int(c.dim),
                'friction': c.friction.tolist(), 'solref': c.solref.tolist(),
                'solimp': c.solimp.tolist(), 'distance_m': float(c.dist),
                'force_contact_frame': force.tolist()})
        samples.append({'index': index, 'time_s': float(data.time),
            'food_lift_m': float(data.xpos[food, 2] - z0[food]),
            'bucket_lift_m': float(data.xpos[bucket, 2] - z0[bucket]),
            'food_hand_contacts': counts[food], 'bucket_hand_contacts': counts[bucket],
            'arm_ctrl_clipped_count': int(np.sum(np.abs(data.ctrl[arm_act]) >
                                                   model.actuator_ctrlrange[arm_act, 1])),
            'hand_force_at_limit_count': int(np.sum(np.abs(data.actuator_force[hand_act]) >=
                                                     model.actuator_forcerange[hand_act, 1] * .999)),
            'arm_target_error_m': float(np.linalg.norm(data.site_xpos[site] - data.mocap_pos[mocap]))})
    out['sampled_reconstruction'] = {'stride': args.sample_stride,
        'caveat': 'mj_forward recomputes contacts and forces at saved states; not original force telemetry.',
        'samples': samples, 'hand_object_contacts': contact_rows,
        'penetration_m': summary([max(0., -c['distance_m']) for c in contact_rows]),
        'normal_force_N': summary([c['force_contact_frame'][0] for c in contact_rows])}
    print('Sampled contacts', out['sampled_reconstruction']['penetration_m'], flush=True)

    # Compare the current in-memory build and quantify visual mesh inertia.
    spec = build_spec(reader.metadata['arena'], hand='shadow')
    current = spec.compile()
    out['current_build_comparison'] = {
        'array_differences': [key for key in arrays if not np.array_equal(
            getattr(model, key), getattr(current, key))],
        'option_differences': {key: [value, plain(getattr(current.opt, key))]
                               for key, value in options.items()
                               if value != plain(getattr(current.opt, key))}}
    for geom in spec.body('boxed_food_0').geoms:
        if geom.contype == 0 and geom.conaffinity == 0:
            geom.mass = 0
            geom.density = 0
    mass_corrected = spec.compile()
    out['food_visual_inertia_check'] = {
        'original_mass_kg': float(current.body_mass[food]),
        'visual_zero_mass_kg': float(mass_corrected.body('boxed_food_0').mass[0]),
        'original_inertia_kg_m2': current.body_inertia[food].tolist(),
        'visual_zero_inertia_kg_m2': mass_corrected.body('boxed_food_0').inertia.tolist()}
    del current, mass_corrected, spec

    # Free flight removes contact as a confound. This checks gravity/integration,
    # not physical mass calibration (free-fall acceleration is mass independent).
    original_dt = float(model.opt.timestep)
    free = model.joint('boxed_food_0_freejoint')
    qa, va = int(free.qposadr[0]), int(free.dofadr[0])
    falls = []
    for dt in (original_dt, original_dt / 2, original_dt / 4):
        model.opt.timestep = dt
        data = reader.restore(0)
        data.qpos[qa:qa + 3] = [0, 0, 3]
        data.qvel[va:va + 6] = 0
        mujoco.mj_forward(model, data)
        acceleration = float(data.qacc[va + 2])
        contacts = 0
        for _ in range(round(.1 / dt)):
            mujoco.mj_step(model, data)
            contacts += sum(food in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
                            for c in data.contact)
        falls.append({'dt_s': dt, 'duration_s': .1, 'initial_acceleration_z_m_s2': acceleration,
                      'drop_m': float(3 - data.qpos[qa + 2]),
                      'analytic_drop_m': .5 * 9.81 * .1**2,
                      'velocity_z_m_s': float(data.qvel[va + 2]),
                      'food_contact_count': int(contacts)})
    model.opt.timestep = original_dt
    out['free_fall'] = falls

    # Choose elevated recorded grasps, then hold the recorded finger commands
    # and arm target. Recompute the original arm feedback each integration step.
    # These local continuations are NOT a repeat of the full human-operated task.
    anchors = {}
    for label in ('food', 'bucket'):
        eligible = [s for s in samples if s[label + '_hand_contacts'] >= 2]
        if eligible:
            anchors[label] = max(eligible, key=lambda s: s[label + '_lift_m'])['index']
    variants = ('baseline', 'noslip_0', 'impratio_1', 'torsion_0.005',
                'sliding_0.5', 'dt_half', 'implicitfast', 'arm_damping_4', 'open_hand')
    friction = model.geom_friction.copy()
    integrator = model.opt.integrator
    holds = []
    for label, index in anchors.items():
        for variant in variants:
            model.opt.timestep = original_dt / (2 if variant == 'dt_half' else 1)
            model.opt.noslip_iterations = 0 if variant == 'noslip_0' else options['noslip_iterations']
            model.opt.impratio = 1 if variant == 'impratio_1' else options['impratio']
            model.opt.integrator = (mujoco.mjtIntegrator.mjINT_IMPLICITFAST
                                    if variant == 'implicitfast' else integrator)
            model.geom_friction[:] = friction
            if variant == 'torsion_0.005':
                model.geom_friction[hand_geoms, 1] = .005
            if variant == 'sliding_0.5':
                # Contact mixing takes the maximum: reduce both contacting surfaces.
                model.geom_friction[:, 0] = np.minimum(model.geom_friction[:, 0], .5)
            data = reader.restore(index)
            for warning in data.warning:
                warning.number = 0
            ctrl = data.ctrl[hand_act].copy()
            if variant == 'open_hand':
                ctrl[:] = np.clip(0, model.actuator_ctrlrange[hand_act, 0],
                                 model.actuator_ctrlrange[hand_act, 1])
            positions = data.xpos[[food, bucket]].copy()
            target_body = food if label == 'food' else handle
            relative = data.xmat[palm].reshape(3, 3).T @ (data.xpos[target_body] - data.xpos[palm])
            start_time = float(data.time)
            max_penetration, max_arm_error = 0., 0.
            steps = round(args.hold_seconds / model.opt.timestep)
            completed = 0
            for _ in range(steps):
                data.ctrl[hand_act] = ctrl
                data.ctrl[arm_act] = opspace(model, data, site, arm_dof,
                    pos=data.mocap_pos[mocap], ori=data.mocap_quat[mocap], joint=home,
                    pos_gains=(400, 400, 400), damping_ratio=4 if variant == 'arm_damping_4' else 1,
                    gravity_comp=True)
                previous_time = float(data.time)
                mujoco.mj_step(model, data)
                completed += 1
                if data.time <= previous_time or not np.isfinite(data.qpos).all():
                    break
                max_penetration = max(max_penetration, max((max(0., -c.dist) for c in data.contact
                    if target_body in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])
                    and (model.geom_bodyid[c.geom1] in hand_bodies or
                         model.geom_bodyid[c.geom2] in hand_bodies)), default=0.))
                max_arm_error = max(max_arm_error, float(np.linalg.norm(
                    data.site_xpos[site] - data.mocap_pos[mocap])))
            mujoco.mj_forward(model, data)
            final_relative = data.xmat[palm].reshape(3, 3).T @ (data.xpos[target_body] - data.xpos[palm])
            row = {'anchor': label, 'index': index, 'start_sim_time_s': start_time,
                   'variant': variant, 'completed_steps': completed, 'requested_steps': steps,
                   'duration_sim_s': float(data.time - start_time),
                   'food_delta_z_m': float(data.xpos[food, 2] - positions[0, 2]),
                   'bucket_delta_z_m': float(data.xpos[bucket, 2] - positions[1, 2]),
                   'object_displacement_in_palm_frame_m': float(np.linalg.norm(final_relative - relative)),
                   'max_hand_object_penetration_m': float(max_penetration),
                   'max_arm_target_error_m': max_arm_error,
                   'final_hand_target_contacts': sum(target_body in (model.geom_bodyid[c.geom1],
                        model.geom_bodyid[c.geom2]) and bool({int(model.geom_bodyid[c.geom1]),
                        int(model.geom_bodyid[c.geom2])} & hand_bodies) for c in data.contact),
                   'warning_counts': [int(w.number) for w in data.warning]}
            holds.append(row)
            print('Hold', label, variant, 'relative displacement',
                  round(row['object_displacement_in_palm_frame_m'], 6), flush=True)
    out['counterfactual_holds'] = {'duration_s': args.hold_seconds,
        'method': 'Fixed recorded hand controls and mocap; original arm feedback recomputed each step.',
        'limitations': 'Two local saved grasps, not full task repetitions or hardware validation. '
                      'Relative displacement includes compliant settling and is not a pure slip measurement. '
                      'dt_half also changes effective refsafe contact stiffness.',
        'anchors': anchors, 'results': holds}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.output)
    print('Wrote', args.output, flush=True)


if __name__ == '__main__':
    main()
