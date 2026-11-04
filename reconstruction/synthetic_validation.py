"""Synthetic-only reference pipeline. There is no real-data execution mode.

Generates analytic box/sphere observations, carves a fixed camera subset, scores
separately held-out cameras, and exports solid occupancy and voxel-centre PLY.
Real image/mask paths are neither accepted nor read by this entrypoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc

import numpy as np

from .cameras import Camera, camera_rays
from .evaluation import Observation, ViewPartition, evaluate_silhouettes, runtime_record
from .figures import comparison_markdown, plot_camera_metrics
from .io import export_ply, save_occupancy
from .rendering import render_occupancy
from .visual_hull import carve_voxels


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_config(config):
    if config.get('data_kind') != 'synthetic' or config.get('real_data_execution') is not False:
        raise ValueError('Real reconstruction is disabled; this entrypoint accepts synthetic primitives only')
    if config.get('schema_version') != 1 or set(config['primitives']) - {'box','sphere'} or not config['primitives']:
        raise ValueError('Only analytic box and sphere fixtures are supported')
    values=config['resolutions']
    if not values or any(isinstance(n,bool) or not isinstance(n,int) or not 4<=n<=96 for n in values):
        raise ValueError('Bounded CPU resolution must be 4 to 96')
    if (isinstance(config['image_size'],bool) or not isinstance(config['image_size'],int)
            or not 17<=config['image_size']<=257):
        raise ValueError('Bounded synthetic image size required')
    for key in ('focal_pixels','camera_radius','primitive_half_extent_or_radius'):
        if not isinstance(config[key],(float,int)) or isinstance(config[key],bool) or not np.isfinite(config[key]) or config[key]<=0:
            raise ValueError('Positive finite synthetic dimensions required')
    inputs,heldout=config['input_camera_indices'],config['heldout_camera_indices']
    if (not inputs or not heldout or any(isinstance(i,bool) or not isinstance(i,int) for i in inputs+heldout)
            or len(inputs+heldout)!=8 or set(inputs)&set(heldout) or set(inputs+heldout)!=set(range(8))):
        raise ValueError('Fixed disjoint input/held-out partition of all eight virtual cameras required')


def virtual_cameras(config):
    cameras={};size=config['image_size'];f=config['focal_pixels']
    for i in range(8):
        angle=i*np.pi/4
        position=np.array([config['camera_radius']*np.sin(angle),config['camera_y'],config['camera_radius']*np.cos(angle)])
        forward=-position/np.linalg.norm(position);up=np.array([0.,-1.,0.])
        right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right)
        transform=np.eye(4);transform[:3,:3]=np.stack((right,down,forward));transform[:3,3]=-transform[:3,:3]@position
        k=np.array([[f,0,(size-1)/2],[0,f,(size-1)/2],[0,0,1.]])
        name=f'V{i:02d}';cameras[name]=Camera(name,k,transform,(size,size),coordinate_frame='synthetic_world',units='synthetic_length_unit')
    return cameras


def analytic_mask(camera, primitive, extent):
    """Ray/primitive intersections independent of voxel carving and DDA."""
    origin,directions=camera_rays(camera)
    if primitive=='sphere':
        a=np.sum(directions*directions,axis=1);b=2*directions@origin;c=origin@origin-extent**2
        discriminant=b*b-4*a*c
        far=(-b+np.sqrt(np.maximum(discriminant,0)))/(2*a)
        hit=(discriminant>0)&(far>0)
    elif primitive=='box':
        near=np.zeros(len(directions));far=np.full(len(directions),np.inf);valid=np.ones(len(directions),bool)
        for axis in range(3):
            parallel=np.abs(directions[:,axis])<1e-14
            valid &= ~parallel | ((origin[axis]>=-extent)&(origin[axis]<extent))
            t0=np.full(len(directions),-np.inf);t1=np.full(len(directions),np.inf)
            moving=~parallel
            a=(-extent-origin[axis])/directions[moving,axis];b=(extent-origin[axis])/directions[moving,axis]
            t0[moving]=np.minimum(a,b);t1[moving]=np.maximum(a,b)
            near=np.maximum(near,t0);far=np.minimum(far,t1)
        hit=valid&(far>near)
    else:
        raise ValueError('Only known synthetic primitives allowed')
    return hit.reshape(camera.image_size[::-1])


def render_figure(grid,cameras,masks,partition,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(11,6))
    for ax,(key,camera) in zip(axes.flat,cameras.items()):
        reference=masks[key];prediction=render_occupancy(grid,camera)
        rgb=np.zeros((*reference.shape,3),np.uint8)+245
        rgb[reference]=[0,190,220];rgb[prediction]=[235,130,25];rgb[reference&prediction]=[80,95,105]
        ax.imshow(rgb);ax.axis('off');ax.set_title(key+(' held out' if key in partition.heldout_camera_ids else ' input'),fontsize=10)
    fig.suptitle(title+'\nCyan: analytic reference; orange: occupied-volume rendering; gray: overlap',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(path,dpi=180);plt.close(fig)


def point_figure(grid,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    points=grid.centers();points=points[::max(1,len(points)//2500)]
    fig=plt.figure(figsize=(6,5));ax=fig.add_subplot(111,projection='3d')
    ax.scatter(*points.T,s=2,c=points[:,1],cmap='viridis',alpha=.45)
    ax.set(xlabel='x (synthetic unit)',ylabel='y',zlabel='z',title=title+'\nVoxel centres; not a surface or silhouette')
    ax.set_box_aspect((1,1,1));fig.tight_layout();fig.savefig(path,dpi=180,bbox_inches='tight',pad_inches=.25);plt.close(fig)


def run(config_path,output,figure_output):
    config=json.loads(Path(config_path).read_text());validate_config(config)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    figure_output=Path(figure_output);figure_output.mkdir(parents=True,exist_ok=True)
    cameras=virtual_cameras(config)
    partition=ViewPartition(tuple(f'V{i:02d}' for i in config['input_camera_indices']),
        tuple(f'V{i:02d}' for i in config['heldout_camera_indices']),tuple(cameras))
    report={'schema_version':1,'data_kind':'synthetic','real_data_execution':False,'real_pilot_ready':False,
        'config_sha256':sha(config_path),'config':config,'python':sys.version,'numpy':np.__version__,
        'platform':platform.platform(),'source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        'camera_calibration':{key:value.to_dict() for key,value in cameras.items()},'cases':[]}
    named={}
    for primitive in config['primitives']:
        extent=config['primitive_half_extent_or_radius']
        masks={key:analytic_mask(camera,primitive,extent) for key,camera in cameras.items()}
        for resolution in config['resolutions']:
            name=f'{primitive}_{resolution}';case_dir=output/name;case_dir.mkdir()
            tracemalloc.start();started=time.perf_counter()
            grid=carve_voxels(cameras,{key:masks[key] for key in partition.input_camera_ids},config['bounds_xyz'],resolution,partition.input_camera_ids)
            observations=[Observation(camera,masks[key],'independent_verified',hashlib.sha256(masks[key].tobytes()).hexdigest(),
                {'kind':'independent_dataset_annotation','data_kind':'synthetic','verified_independent':True,
                 'derived_from_prediction':False,'source_id':f'analytic_{primitive}_{key}',
                 'reviewer':'analytic primitive generator, not a human mask annotation',
                 'evidence':'Known virtual primitive intersected by camera rays; masks are not generated from the carved voxels.'}) for key,camera in cameras.items()]
            scores=evaluate_silhouettes(grid,observations,partition,boundary_tolerance_px=config['boundary_tolerance_px'])
            export_ply(grid,case_dir/'voxel_centres.ply',metadata={'data_kind':'synthetic','real_data_execution':False})
            save_occupancy(grid,case_dir/'solid_occupancy.npz')
            elapsed=time.perf_counter()-started;_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
            points=grid.centers();truth_volume=(2*extent)**3 if primitive=='box' else 4*np.pi*extent**3/3
            volume=float(grid.occupancy.sum()*np.prod(grid.voxel_size_xyz))
            if primitive=='sphere':
                inside=np.linalg.norm(points,axis=1)<extent
            else:
                inside=np.all(np.abs(points)<extent,axis=1)
            record={'name':name,'primitive':primitive,'resolution':resolution,'data_kind':'synthetic',
                'occupied_voxels':int(grid.occupancy.sum()),'analytic_primitive_volume':truth_volume,
                'occupied_volume':volume,'relative_volume_difference':(volume-truth_volume)/truth_volume,
                'fraction_retained_centres_inside_true_primitive':float(inside.mean()),'scores':scores,
                'resource':runtime_record(status='completed',elapsed_seconds=elapsed,peak_memory_bytes=peak,
                    memory_measurement='Python tracemalloc peak during carve/evaluation/export; not process RSS or GPU memory',
                    output_bytes=sum(p.stat().st_size for p in case_dir.iterdir() if p.is_file()))}
            (case_dir/'case.json').write_text(json.dumps(record,indent=2)+'\n');report['cases'].append(record);named[name]=scores
            render_figure(grid,cameras,masks,partition,figure_output/f'{name}_silhouettes.png',f'SYNTHETIC {primitive}, {resolution}^3 voxels')
            plot_camera_metrics(scores,figure_output/f'{name}_iou.png',title=f'SYNTHETIC {primitive}, {resolution}^3: analytic-reference IoU')
            if resolution==max(config['resolutions']):
                point_figure(grid,figure_output/f'{name}_points.png',f'SYNTHETIC {primitive}, {resolution}^3')
    report['figure_sha256']={p.name:sha(p) for p in figure_output.glob('*.png')}
    (figure_output.parent/'synthetic_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (figure_output.parent/'SYNTHETIC_RESULTS.md').write_text('# Synthetic implementation validation\n\nKnown virtual primitives only; no Assembly101 or learned reconstruction result.\n\n'+comparison_markdown(named))
    print(json.dumps({'cases':len(report['cases']),'results':str(figure_output.parent/'synthetic_validation.json'),'real_data_execution':False}))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/reconstruction_synthetic.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--figures',required=True)
    args=parser.parse_args();run(args.config,args.output,args.figures)


if __name__=='__main__':
    main()
