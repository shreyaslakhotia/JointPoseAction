"""Lightweight, hash-checked source and correspondence figures before inference.

Input directory holds copies of measured reports and native PNGs, arranged by
recording ID. No image content, calibration, matches or scientific gates change.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def font(size):
    try:
        return ImageFont.truetype('DejaVuSans.ttf',size)
    except OSError:
        return ImageFont.truetype('arial.ttf',size)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.inputs);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    survey=json.loads((root/'survey.json').read_text())
    geometry=json.loads((root/'geometry-268824.json').read_text())
    manifest={'purpose':'unvalidated_multiview_candidate_review','multiview_dataset_ready':False,
              'source_survey_sha256':sha(root/'survey.json'),'geometry_sha256':sha(root/'geometry-268824.json'),
              'script_sha256':sha(__file__),'figures':[]}
    for number,rec in enumerate(survey['recordings'],1):
        sheet=Image.new('RGB',(1920,660),'white');d=ImageDraw.Draw(sheet)
        d.text((12,5),f'Candidate {number:02d} | RAW source PTS 0 | synchronization UNVERIFIED | diagnostic review only',fill='black',font=font(18))
        sources=[]
        for i,view in enumerate(rec['views']):
            frame=next(f for f in view['frames'] if f['pts_seconds']==0)
            source=root/rec['recording_id']/Path(frame['path']).name
            if sha(source)!=frame['sha256']:
                raise ValueError('Source frame hash mismatch')
            im=Image.open(source).convert('RGB');im.thumbnail((480,270))
            x=i%4*480;y=35+i//4*305;sheet.paste(im,(x,y+28))
            d.text((x+8,y+4),view['camera_id']+' | 1920 x 1080 native',fill='black',font=font(16))
            sources.append({'camera_id':view['camera_id'],'frame_sha256':frame['sha256'],'pts':frame['pts'],
                            'time_base':[frame['time_base_numerator'],frame['time_base_denominator']]})
        path=out/f'candidate{number:02d}_raw_eight_camera.jpg';sheet.save(path,quality=92)
        manifest['figures'].append({'path':path.name,'sha256':sha(path),'recording_id':rec['recording_id'],
            'sources':sources,'changes':'Downsampling and external labels only. No geometric rectification, timing alignment or pixel retouching.'})
    # A high-residual pair and a low-residual pair are both retained. The visual
    # patches are for independent review, never selected by the known-F residual.
    rec=geometry['recordings'][0]
    for camera in ('C10390','C10404'):
        pair=next(r for r in rec['pairs'] if r['camera_ids'][1]==camera)
        sources=[root/rec['recording_id']/Path(f).name for f in pair['source_image_paths']]
        if [sha(f) for f in sources]!=pair['source_image_sha256']:
            raise ValueError('Geometry image hash mismatch')
        images=[Image.open(f).convert('RGB') for f in sources]
        candidates=[(i,c) for i,c in enumerate(pair['correspondences']) if c['image_ransac_inlier']][:12]
        sheet=Image.new('RGB',(1040,80+len(candidates)*145),'white');d=ImageDraw.Draw(sheet)
        d.text((8,8),f'C10119 / {camera} | image-only RANSAC candidates | NOT verified correspondences',fill='black',font=font(17))
        d.text((8,36),'First 12 by stored match order; no selection by calibration residual. Source PTS 1 s; sync unverified.',fill='black',font=font(15))
        for row,(index,candidate) in enumerate(candidates):
            y=80+row*145
            d.text((8,y+12),f'Match {index}',fill='black',font=font(15))
            for side in range(2):
                px,py=candidate[f'raw_pixel_{side+1}'];x0,y0=round(px)-48,round(py)-48
                patch=images[side].crop((x0,y0,x0+96,y0+96)).resize((128,128))
                pd=ImageDraw.Draw(patch);cx=(px-x0)*128/96;cy=(py-y0)*128/96
                pd.ellipse((cx-5,cy-5,cx+5,cy+5),outline='yellow',width=2)
                x=145+side*440;sheet.paste(patch,(x,y));
                d.text((x+145,y+15),f'{pair["camera_ids"][side]}\n({px:.2f}, {py:.2f})',fill='black',font=font(15))
        path=out/f'green01_C10119_{camera}_candidate_patches.jpg';sheet.save(path,quality=94)
        manifest['figures'].append({'path':path.name,'sha256':sha(path),'recording_id':rec['recording_id'],
            'source_image_sha256':pair['source_image_sha256'],'correspondence_indices':[i for i,_ in candidates],
            'changes':'96 x 96 native patches enlarged to 128 x 128 for inspection; yellow location markers. Original photos unchanged.',
            'independently_verified_correspondences':False})
    (out/'source_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'figures':len(manifest['figures']),'output':str(out),'multiview_dataset_ready':False}))


if __name__=='__main__':
    main()
