import numpy as np

from forecasting.scene_features import SCENE_FEATURE_DIM, extract_scene_features


class FakeObject:
    def __init__(self, location, label, isbox=True):
        self.location = np.asarray(location, dtype=np.float32)
        self.label = np.asarray(label, dtype=np.float32)
        self.isbox = isbox


class FakeKitchen:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def get_environment(self, frame, ignore_oob=True, use_pointcloud=False):
        self.calls.append((frame, ignore_oob, use_pointcloud))
        return self.objects


def _label(index):
    out = np.zeros(13, dtype=np.float32)
    out[index] = 1.0
    return out


def test_scene_features_zero_without_kitchen():
    pose = np.ones((29, 3), dtype=np.float32)
    features = extract_scene_features(None, frame=10, pose3d=pose)
    assert features.shape == (SCENE_FEATURE_DIM,)
    assert features.dtype == np.float32
    assert np.allclose(features, 0.0)


def test_scene_features_selects_nearest_box_by_xy_distance():
    pose = np.zeros((29, 3), dtype=np.float32)
    far_box = np.array(
        [
            [5.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 1.0, 0.0],
            [5.0, 1.0, 0.0],
            [5.0, 0.0, 1.0],
            [6.0, 0.0, 1.0],
            [6.0, 1.0, 1.0],
            [5.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    near_box = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    kitchen = FakeKitchen(
        [
            FakeObject(far_box, _label(3), isbox=True),
            FakeObject(near_box, _label(10), isbox=True),
        ]
    )

    features = extract_scene_features(kitchen, frame=42, pose3d=pose)

    assert kitchen.calls == [(42, True, False)]
    assert np.allclose(features[:13], _label(10))
    assert np.allclose(features[13:17], [0.15, 0.05, np.sqrt(2.5) / 10.0, 1.0])


def test_scene_features_handles_cylinder_location():
    pose = np.zeros((29, 3), dtype=np.float32)
    kitchen = FakeKitchen([FakeObject([0.0, 3.0, 0.0, 0.4], _label(4), isbox=False)])

    features = extract_scene_features(kitchen, frame=7, pose3d=pose)

    assert np.allclose(features[:13], _label(4))
    assert np.allclose(features[13:17], [0.0, 0.3, 0.3, 1.0])
    assert np.isfinite(features).all()
