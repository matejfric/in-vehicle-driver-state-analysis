from pathlib import Path

from model.common import Anomalies


def test_anomalies() -> None:
    DATASET_NAME = '2024-10-28-driver-all-frames/2021_08_31_geordi_enyaq'
    DATASET_DIR = Path().home() / f'source/driver-dataset/{DATASET_NAME}'
    ANOMALIES_FILE = DATASET_DIR / 'anomal' / 'labels.txt'
    assert ANOMALIES_FILE.exists(), f'Anomalies file does not exist: {ANOMALIES_FILE}'

    anomalies = Anomalies.from_file(ANOMALIES_FILE)

    n_test_frames = len(list((ANOMALIES_FILE.parent / 'images').glob('*.jpg')))
    y_true = anomalies.to_ground_truth(n_test_frames)

    # Calculate the expected sum of anomaly durations
    expected_sum = sum([anomaly.end - anomaly.start for anomaly in anomalies])

    # Assert that the sum of y_true matches the expected anomaly duration sum
    assert (
        sum(y_true) == expected_sum
    ), 'Mismatch in ground truth and anomaly duration sums'
