"""The review package must not turn a high score into ground-truth acceptance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from segmentation_pilot.review import render
from segmentation_pilot.sam3_runner import save_stage


class ReviewTests(unittest.TestCase):
    def test_native_stage_pixels_preserved_and_high_score_is_not_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();image_id='test_C10095_pts0'
            frame_dir=run/image_id;frame_dir.mkdir()
            source=frame_dir/'frame.png';Image.new('RGB',(32,24),(100,110,120)).save(source)
            before={}
            for stage,shift in [('text_only',0),('refined',5)]:
                masks=np.zeros((2,24,32),dtype=bool)
                masks[0,2:7,2:7]=True
                masks[1,5+shift:12+shift,10:20]=True
                masks[1,8+shift,15]=False  # Native hole must remain a hole.
                save_stage(frame_dir,stage,masks,np.array([[2,2,7,7],[10,5+shift,20,12+shift]]),np.array([.1,.9]),
                    height=24,width=32,prompt='toy',diagnostic_2d_review=True,
                    correspondence_scope='diagnostic_2d_review',temporal_sync_verified=False)
                before[stage]=(frame_dir/stage/'native_masks.npz').read_bytes()
            record={'image_id':image_id,'frame_path':f'{image_id}/frame.png',
                'frame_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'width':32,'height':24,'object_id':'synthetic','recording_id':'recording',
                'camera_id':'C10095','source_pts_seconds':0.,'refinement':{'box_xyxy':[5,5,25,20]}}
            (run/'completed.json').write_text(json.dumps({'purpose':'diagnostic_2d_review',
                'multiview_dataset_ready':False,'accuracy_established':False,'images':[record]}))
            (run/'provenance.json').write_text(json.dumps({'checkpoint_sha256':'synthetic_not_a_real_model'}))
            output=root/'review';render(run,output)
            result=json.loads((output/'review_manifest.json').read_text())
            self.assertFalse(result['accuracy_established'])
            self.assertFalse(result['multiview_dataset_ready'])
            self.assertIsNone(result['reference_accuracy_metrics'])
            for row in result['per_view_metrics']:
                self.assertEqual(row['display_candidate_id'],1)
                self.assertFalse(row['correct_physical_instance_verified'])
                self.assertIsNone(row['iou'])
            for stage in before:
                self.assertEqual(before[stage],(frame_dir/stage/'native_masks.npz').read_bytes())
                mask=np.asarray(Image.open(output/'native_display_examples'/image_id/stage/'display_mask.png'))
                shift=0 if stage=='text_only' else 5
                self.assertEqual(mask[8+shift,15],0)
                self.assertEqual(mask[7+shift,15],255)
            self.assertFalse(np.array_equal(
                Image.open(output/'native_display_examples'/image_id/'text_only/display_mask.png'),
                Image.open(output/'native_display_examples'/image_id/'refined/display_mask.png')))


if __name__=='__main__':
    unittest.main()
