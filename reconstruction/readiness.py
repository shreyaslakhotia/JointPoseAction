"""Audit evidence for a later real run; real execution is unavailable this phase.

An accepted JSON attestation is not physical proof. Its reviewer must actually
inspect the native masks and independent calibration/correspondence evidence.
The current synthetic-only entrypoint cannot be released by editing a boolean.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

REQUIRED_CHECKS = ('native_mask_review', 'calibration_accuracy',
                   'temporal_or_static_correspondence', 'coordinate_frame_and_scale',
                   'source_integrity')


def audit_pilot(pilot: dict, *, evidence_root: Path) -> dict:
    for field in ('recording_id','object_id'):
        if not isinstance(pilot.get(field),str) or not pilot[field]:
            raise ValueError('Exact recording/object identity required')
    cameras=pilot.get('camera_ids')
    if (not isinstance(cameras,list) or not cameras or any(not isinstance(c,str) or not c for c in cameras)
            or len(cameras)!=len(set(cameras))):
        raise ValueError('Explicit unique proposed camera IDs required')
    checks=pilot.get('checks',{});results={}
    for name in REQUIRED_CHECKS:
        row=checks.get(name,{})
        reason='Review pending; no approval inferred from inference completion'
        passed=False
        if row.get('status')=='accepted':
            try:
                path=Path(row['evidence_path'])
                if not path.is_absolute():
                    path=Path(evidence_root)/path
                expected=row['evidence_sha256']
                if not isinstance(expected,str) or not re.fullmatch('[0-9a-f]{64}',expected):
                    raise ValueError('Exact evidence SHA-256 required')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                    raise ValueError('Evidence file SHA-256 mismatch')
                evidence=json.loads(path.read_text())
                if (evidence.get('check')!=name or evidence.get('status')!='accepted'
                        or evidence.get('recording_id')!=pilot['recording_id']
                        or evidence.get('object_id')!=pilot['object_id']
                        or evidence.get('camera_ids')!=cameras):
                    raise ValueError('Evidence scope/identity/camera inventory mismatch')
                if any(not isinstance(evidence.get(k),str) or not evidence[k].strip()
                       for k in ('reviewer','review_reference','limitations')):
                    raise ValueError('Named review, approval reference and limitations required')
                if evidence.get('diagnostic_2d_review') is True or evidence.get('multiview_dataset_ready') is False:
                    raise ValueError('Diagnostic output cannot serve as scientific approval')
                if name=='native_mask_review' and (evidence.get('native_masks_inspected') is not True
                        or evidence.get('physical_instances_approved') is not True):
                    raise ValueError('Native masks and physical instances require actual review')
                if name=='calibration_accuracy' and evidence.get('independent_geometric_evidence_verified') is not True:
                    raise ValueError('Schema/provenance checks alone do not verify camera accuracy')
                if name=='temporal_or_static_correspondence' and evidence.get('scope') not in ('synchronized_interval','verified_static_state_only'):
                    raise ValueError('Measured temporal mapping or explicitly verified static state required')
                if name=='coordinate_frame_and_scale':
                    metric=(evidence.get('scale_evidence_verified') is True and evidence.get('units') not in ('unknown','unverified'))
                    nonmetric=(evidence.get('nonmetric_shape_scope_only') is True and evidence.get('metric_claims_allowed') is False)
                    if (not evidence.get('coordinate_frame') or not evidence.get('units')
                            or evidence.get('coordinate_conventions_verified') is not True or not (metric or nonmetric)):
                        raise ValueError('Verified coordinate conventions and scale evidence or an explicit nonmetric restriction required')
                passed,reason=True,'Hashed accepted review recorded; physical validity remains the reviewer responsibility'
            except (OSError,ValueError,KeyError,TypeError) as error:
                reason=str(error)
        results[name]={'passed':passed,'reason':reason}
    return {'recording_id':pilot['recording_id'],'object_id':pilot['object_id'],
        'checks':results,'data_checks_passed':all(r['passed'] for r in results.values()),
        'real_execution_allowed':False,'phase':'implementation_and_synthetic_validation_only',
        'execution_blocker':'No real reconstruction or learned-model inference entrypoint is enabled in this phase.'}


def audit_manifest(path):
    path=Path(path)
    payload=json.loads(path.read_text())
    if payload.get('schema_version')!=1 or not isinstance(payload.get('pilots'),list):
        raise ValueError('Expected schema_version 1 with explicit pilot list')
    return [audit_pilot(p,evidence_root=path.parent) for p in payload['pilots']]
