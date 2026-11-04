"""Render diagnostic native-mask predictions without declaring an instance correct.

The highest-confidence query is a *display convention*, not target selection.
All other queries remain in the inference output. No reference masks, accuracy,
silhouette union, reconstruction or geometric acceptance are created here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .sam3_runner import unpack_native_masks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size=15):
    try:
        return ImageFont.truetype('DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def thumbnail(image, size):
    canvas = Image.new('RGB', size, 'white')
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return canvas


def overlay(rgb, mask):
    base = np.asarray(rgb).copy()
    base[mask] = (base[mask].astype(float)*.55 + np.array([0,210,240])*.45).astype(np.uint8)
    return Image.fromarray(base)


def render(run, output, max_bytes=536870912):
    run, output = Path(run).resolve(), Path(output).resolve()
    completed = json.loads((run/'completed.json').read_text())
    if (completed.get('purpose') != 'diagnostic_2d_review'
            or completed.get('multiview_dataset_ready') is not False
            or completed.get('accuracy_established') is not False):
        raise ValueError('This renderer only accepts explicitly diagnostic completed inference')
    provenance = json.loads((run/'provenance.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    rows, figures, review_decisions = [], [], []
    objects = sorted({row['object_id'] for row in completed['images']})
    for object_index, object_id in enumerate(objects, 1):
        images = sorted([row for row in completed['images'] if row['object_id'] == object_id], key=lambda r:r['camera_id'])
        sheets = {name: Image.new('RGB',(1920,650),'white') for name in
                  ('original','text_only_overlay','text_only_mask','refined_overlay','refined_mask')}
        comparison = Image.new('RGB',(1200,8*310+75),'white')
        title = f'{object_id} | DIAGNOSTIC ONLY | highest-score query displayed; correct-instance review pending'
        for sheet in sheets.values():
            ImageDraw.Draw(sheet).text((10,5),title,fill='black',font=font())
        ImageDraw.Draw(comparison).text((10,5),title,fill='black',font=font(14))
        for col,label in enumerate(('Original target region','Text-only highest score','Box-refined highest score')):
            ImageDraw.Draw(comparison).text((col*400+10,35),label,fill='black',font=font())
        for view_index, row in enumerate(images):
            source = run/row['frame_path']
            if sha(source) != row['frame_sha256']:
                raise ValueError('Saved frame SHA-256 mismatch')
            rgb = Image.open(source).convert('RGB')
            if rgb.size != (row['width'],row['height']):
                raise ValueError('Native image resolution mismatch')
            x,y = view_index%4*480, 35+view_index//4*305
            sheets['original'].paste(thumbnail(rgb,(480,270)),(x,y+25))
            ImageDraw.Draw(sheets['original']).text((x+8,y+3),row['camera_id']+' raw PTS '+str(row['source_pts_seconds']),fill='black',font=font())
            box = row['refinement']['box_xyxy']
            crop = (max(0,int(box[0])-50), max(0,int(box[1])-50), min(rgb.width,int(box[2])+50), min(rgb.height,int(box[3])+50))
            comp_y = 75+view_index*310
            comparison.paste(thumbnail(rgb.crop(crop),(400,280)),(0,comp_y+25))
            ImageDraw.Draw(comparison).text((8,comp_y+3),row['camera_id'],fill='black',font=font())
            for stage_index, stage_name in enumerate(('text_only','refined'),1):
                stage = run/row['image_id']/stage_name
                info = json.loads((stage/'candidates.json').read_text())
                packed = stage/info['native_masks']['path']
                if sha(packed) != info['native_masks']['sha256']:
                    raise ValueError('Native-mask SHA-256 mismatch')
                candidates = info['candidates']
                masks = unpack_native_masks(packed)
                if len(masks) != len(candidates):
                    raise ValueError('Native-mask candidate index mismatch')
                highest = max(candidates,key=lambda c:c['grounding_confidence']) if candidates else None
                display_id = highest['candidate_id'] if highest else None
                if highest is not None:
                    mask = masks[display_id]
                    colored = overlay(rgb,mask)
                    mono = Image.fromarray(mask.astype(np.uint8)*255).convert('RGB')
                    native_dir = output/'native_display_examples'/row['image_id']/stage_name
                    native_dir.mkdir(parents=True)
                    # Transparent overlay is derived from the exact native binary array.
                    # It is a visualization; native_masks.npz remains the authoritative mask.
                    rgba = np.zeros((rgb.height,rgb.width,4),dtype=np.uint8)
                    rgba[mask] = [0,210,240,115]
                    Image.fromarray(rgba,'RGBA').save(native_dir/'mask_overlay_rgba.png')
                    Image.fromarray(mask.astype(np.uint8)*255).save(native_dir/'display_mask.png')
                    colored.save(native_dir/'overlay.jpg',quality=92)
                    score = highest['grounding_confidence']
                    label = f"{row['camera_id']} q{display_id} score={score:.3f}"
                else:
                    colored, mono = rgb.copy(), Image.new('RGB',rgb.size,'black')
                    score, label = None, row['camera_id']+' NO CANDIDATE'
                for suffix, content in [('overlay',colored),('mask',mono)]:
                    sheet = sheets[stage_name+'_'+suffix]
                    sheet.paste(thumbnail(content,(480,270)),(x,y+25))
                    ImageDraw.Draw(sheet).text((x+8,y+3),label,fill='black',font=font())
                comparison.paste(thumbnail(colored.crop(crop),(400,280)),(stage_index*400,comp_y+25))
                ImageDraw.Draw(comparison).text((stage_index*400+8,comp_y+3),label,fill='black',font=font())
                rows.append({'image_id':row['image_id'],'object_id':object_id,'recording_id':row['recording_id'],
                    'camera_id':row['camera_id'],'stage':stage_name,'source_pts_seconds':row['source_pts_seconds'],
                    'candidate_count':len(candidates),'candidates_score_gt_0_5':sum(c['grounding_confidence']>.5 for c in candidates),
                    'display_candidate_id':display_id,'display_confidence':score,
                    'display_image_coverage_fraction':highest['image_coverage_fraction'] if highest else None,
                    'correct_physical_instance_verified':False,'manual_review_status':'pending',
                    'iou':None,'dice':None,'boundary_f1':None,'accuracy_established':False})
                review_decisions.append({'image_id':row['image_id'],'stage':stage_name,
                    'correct_physical_candidate_id':None,'reviewer':None,'assessment':None,
                    'notes':None,'reference_mask_available':False})
                del masks
        for name, sheet in sheets.items():
            path=output/f'{object_id}_{name}.jpg'
            sheet.save(path,quality=91)
            figures.append({'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size,
                            'caption':title+'; native full frames downsampled for display, no geometry/sync claim.'})
        path=output/f'{object_id}_prompt_comparison.jpg'
        comparison.save(path,quality=92)
        figures.append({'path':path.name,'sha256':sha(path),'bytes':path.stat().st_size,
                        'caption':title+'; identical crop for both stages, crop based on pre-inference manual positive box +50px.'})
        used=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if used>max_bytes:
            raise RuntimeError('Review output byte limit exceeded; stop before further rendering')
    with (output/'per_view_descriptive.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'review_decisions.template.json').write_text(json.dumps({'purpose':'manual_review_not_ground_truth','reviews':review_decisions},indent=2)+'\n')
    manifest={'schema_version':1,'purpose':'diagnostic_2d_review','multiview_dataset_ready':False,
        'temporal_sync_verified':False,'calibration_accuracy_verified':False,'accuracy_established':False,
        'inference_path':str(run),'inference_completed_sha256':sha(run/'completed.json'),
        'inference_provenance_sha256':sha(run/'provenance.json'),'checkpoint_sha256':provenance['checkpoint_sha256'],
        'renderer_sha256':sha(__file__),'figures':figures,'per_view_metrics':rows,
        'display_rule':'Highest grounding score per stage. Display does not establish physical target correctness.',
        'independent_mask_reference_available':False,'reference_accuracy_metrics':None,
        'review_status':'pending_user_visual_review',
        'complete_native_masks_path':str(run),'native_display_examples':'native_display_examples/',
        'manual_interventions':'Pre-inference agent positive boxes, unchanged between mask inference and display.'}
    (output/'review_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'review_path':str(output),'figures':len(figures),'rows':len(rows),'multiview_dataset_ready':False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    render(args.run,args.output)


if __name__=='__main__':
    main()
