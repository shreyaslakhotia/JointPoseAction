"""Bind reviewed prompt boxes to acquired native frames; never accepts a pilot.

The source survey, geometry and tracking files are measured Slurm outputs.
This creates an explicitly diagnostic manifest, not calibration approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--survey', required=True)
    p.add_argument('--geometry', required=True)
    p.add_argument('--stability', required=True)
    p.add_argument('--boxes', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    survey, geometry, stability, boxes = [json.loads(Path(f).read_text()) for f in
                                        (args.survey, args.geometry, args.stability, args.boxes)]
    if geometry['survey_sha256'] != sha(args.survey) or stability['survey_sha256'] != sha(args.survey):
        raise ValueError('Geometry/tracking do not refer to this exact survey')
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=False, parents=True)
    assessment = {'schema_version': 1, 'multiview_dataset_ready': False,
                  'calibration_accuracy_verified': False, 'temporal_sync_verified': False,
                  'geometry_path': str(Path(args.geometry).resolve()), 'geometry_sha256': sha(args.geometry),
                  'recordings': [],
                  'limitations': ['Vendor provenance and matrix structure are verified; geometric accuracy is unresolved.',
                                  'Automatic image correspondences are not independent verified ground truth.',
                                  'Equal raw stream PTS does not establish cross-camera synchronization.',
                                  'Local landmark tracking is not proof of whole-object or cross-view rigidity.',
                                  'No reconstruction or calibrated benchmark acceptance is authorized by this file.']}
    images = []
    for index, obj in enumerate(boxes['objects'], 1):
        rec = next(r for r in survey['recordings'] if r['recording_id'] == obj['recording_id'])
        geo = next(r for r in geometry['recordings'] if r['recording_id'] == obj['recording_id'])
        if not rec['calibration_structural_checks_passed'] or len(rec['views']) != 8:
            raise ValueError('Exact native eight-camera inventory and structural calibration checks required')
        pairs = [{k: row[k] for k in ('camera_ids', 'status', 'mutual_ratio_match_count',
                  'image_fitted_ransac_inlier_count', 'calibration_epipolar_residuals_on_image_ransac_inliers')
                  if k in row} for row in geo['pairs']]
        assessment['recordings'].append({'recording_id': obj['recording_id'],
            'object_id': obj['object_id'], 'calibration_path': rec['calibration_path'],
            'calibration_sha256': rec['calibration_sha256'], 'vendor_calibration_provenance_verified': True,
            'calibration_structure_verified': True, 'calibration_accuracy_verified': False,
            'temporal_sync_verified': False, 'multiview_dataset_ready': False,
            'image_diagnostics': pairs})
        for view in rec['views']:
            frame = next(f for f in view['frames'] if f['pts_seconds'] == boxes['source_pts_seconds'])
            images.append({'image_id': f"pilot{index:02d}_{view['camera_id']}_pts0",
                'path': frame['path'], 'sha256': frame['sha256'], 'source_pts_seconds': frame['pts_seconds'],
                'object_id': obj['object_id'], 'recording_id': rec['recording_id'], 'camera_id': view['camera_id'],
                'prompt': boxes['prompt'], 'object_description_from_rgb': obj['description'],
                'refinement': {'box_xyxy': obj['boxes_xyxy'][view['camera_id']],
                    'reason': 'Select the physical toy; exclude separate screwdriver and printed illustration where separable. Visible surface only; no amodal completion.',
                    'intervention': 'agent_visual_review'}})
    assessment_path = output/'calibration_assessment.json'
    assessment_path.write_text(json.dumps(assessment, indent=2)+'\n')
    manifest = {'schema_version': 1, 'purpose': 'diagnostic_2d_review', 'multiview_dataset_ready': False,
        'calibration_accuracy_verified': False, 'temporal_sync_verified': False,
        'source_survey_path': str(Path(args.survey).resolve()), 'source_survey_sha256': sha(args.survey),
        'calibration_assessment_path': str(assessment_path), 'calibration_assessment_sha256': sha(assessment_path),
        'tracking_stability_path': str(Path(args.stability).resolve()), 'tracking_stability_sha256': sha(args.stability),
        'prompt_boxes_source': str(Path(args.boxes).resolve()), 'prompt_boxes_sha256': sha(args.boxes),
        'images': images}
    path = output/'images.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'manifest': str(path), 'images': len(images), 'multiview_dataset_ready': False}))


if __name__ == '__main__':
    main()
