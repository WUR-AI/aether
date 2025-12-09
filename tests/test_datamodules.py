import pytest

from src.data.base_datamodule import BaseDataModule
from src.data.components.butterfly_dataset import ButterflyDataset


@pytest.mark.parametrize(
    "modalities, target, numericals",
    [
        (['coords'], True, False),
        (['coords'], True, True),
        (['coords'], False, False)
    ]
)
def test_butterfly_dataset(modalities, target, numericals):
    df_path = 'data/model_ready/s2bms_presence_with_aux_data.csv'

    dataset = ButterflyDataset(df_path, modalities, target, numericals)

    assert dataset.modalities == modalities
    assert dataset.target == target
    assert dataset.numericals == numericals

    data_point = dataset[0]

    if target:
        assert dataset.target_names is not None
        assert len(dataset.target_names) > 0
        assert data_point.get('target') is not None
    else:
        assert dataset.target_names is None

    if numericals:
        assert dataset.numerical_names is not None
        assert len(dataset.numerical_names) > 0
        assert data_point.get('numericals') is not None
    else:
        assert dataset.numerical_names is None

    for modality in modalities:
        assert data_point.get('eo', {}).get(modality) is not None

        if modality == 'coords':
            assert len(data_point.get('eo', {}).get(modality)) == 2




@pytest.mark.parametrize(
    "modalities, target, numericals, batch_size",
    [
        (['coords'], True, False, 32),
        (['coords'], True, True, 16),
        (['coords'], False, False, 4)
    ]
)
def test_butterfly_datamodule(modalities, target, numericals, batch_size):
    df_path = 'data/model_ready/s2bms_presence_with_aux_data.csv'

    dataset = ButterflyDataset(df_path, modalities, target, numericals)

    dm = BaseDataModule(dataset, batch_size=batch_size)

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == len(dataset)

    batch = next(iter(dm.train_dataloader()))
    for modality in modalities:
        assert len(batch.get('eo', {}).get(modality)) == batch_size
    if target:
        assert batch.get('target') is not None
        assert len(batch.get('target')) == batch_size
    else:
        assert batch.get('target') is None

    if numericals:
        assert batch.get('numericals') is not None
        assert len(batch.get('numericals')) == batch_size
    else:
        assert batch.get('numericals') is None