import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from reconstruction.readiness import REQUIRED_CHECKS,audit_pilot
from reconstruction.synthetic_validation import validate_config


class ReadinessTests(unittest.TestCase):
    def test_claimed_pass_without_hashed_evidence_does_not_release_run(self):
        pilot={'recording_id':'r','object_id':'o','camera_ids':['c'],
               'checks':{key:{'status':'accepted'} for key in REQUIRED_CHECKS}}
        result=audit_pilot(pilot,evidence_root=Path('.'))
        self.assertFalse(result['data_checks_passed'])
        self.assertFalse(result['real_execution_allowed'])

    def test_even_complete_scoped_attestations_cannot_enable_real_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pilot={'recording_id':'r','object_id':'o','camera_ids':['c'],'checks':{}}
            for key in REQUIRED_CHECKS:
                row={'check':key,'status':'accepted','recording_id':'r','object_id':'o','camera_ids':['c'],
                    'reviewer':'synthetic test fixture','review_reference':'synthetic unit test only',
                    'limitations':'Fixture does not approve any real data.','native_masks_inspected':True,
                    'physical_instances_approved':True,'independent_geometric_evidence_verified':True,
                    'scope':'verified_static_state_only','coordinate_frame':'test','units':'test_units',
                    'coordinate_conventions_verified':True,'scale_evidence_verified':True}
                path=root/(key+'.json');path.write_text(json.dumps(row))
                pilot['checks'][key]={'status':'accepted','evidence_path':path.name,
                                     'evidence_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            result=audit_pilot(pilot,evidence_root=root)
            self.assertTrue(result['data_checks_passed'])
            self.assertFalse(result['real_execution_allowed'])
            path.write_text('{}')
            self.assertFalse(audit_pilot(pilot,evidence_root=root)['data_checks_passed'])

    def test_real_and_diagnostic_data_cannot_enter_synthetic_driver(self):
        for kind in ('real','diagnostic_2d_review'):
            with self.assertRaisesRegex(ValueError,'Real reconstruction is disabled'):
                validate_config({'data_kind':kind,'real_data_execution':True})


if __name__=='__main__':
    unittest.main()
