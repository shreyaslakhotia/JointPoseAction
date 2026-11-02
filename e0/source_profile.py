"""Declared E0 recognition protocol; physical metric calibration is not asserted."""


def dimensionless_profile():
    return {
        "status": "validated_real_sample", "schema": "assembly101_cleaned_landmarks3D",
        "hand_keys": ["0", "1"], "hand_order": ["left", "right"], "wrist_index": 5,
        "joint_order": ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip", "wrist",
                        "thumb_intermediate", "thumb_distal", "index_proximal", "index_intermediate", "index_distal",
                        "middle_proximal", "middle_intermediate", "middle_distal", "ring_proximal", "ring_intermediate",
                        "ring_distal", "pinky_proximal", "pinky_intermediate", "pinky_distal", "palm_center"],
        "units_to_m": None, "feature_units": "dimensionless_wrist_index_length", "scale_joints": [5, 8],
        "physical_units_verified": False, "coordinate_frame": "release_world_axes_preserved_translation_normalized",
        "pose_fps": 60, "annotation_fps": 30, "annotation_index_base": 0, "pose_frame_offset": 0,
        "end_inclusive": True, "zero_is_invalid": False,
        "temporal_protocol": "reference_reader_doubled_frame_ids_inclusive_end",
        "independent_video_clock_verification": False,
        "evidence": [
            "HandFormer 640a10a6bd03f85842a2662e67cf30444ab1d673 Preprocess/1_get_clip_data.py: hand keys, wrist, doubled IDs, end+1",
            "Official Assembly101 paper section 3.3: 21 landmarks in world coordinates",
            "Official MS-G3D graph and producer-family UmeTrack enumeration corroborate joint order",
            "Cluster jobs 242734 and 242754: complete CRC-verified cleaned schema and confidence/transform associations",
            "Known anomalies excluded; every acquired recording must pass companion cadence/key checks",
            "E0 dimensionless scale uses median positive-score wrist-to-index-proximal length within the segment; no dataset-fitted statistics",
            "E3 metric hand/object fusion requires independent unit/frame calibration; E0 asserts no metric calibration"
        ]}
