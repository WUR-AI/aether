from src.data.base_dataset import BaseDataset
from src.data.butterfly_dataset import ButterflyDataset
from src.data.heat_guatemala_dataset import HeatGuatemalaDataset
from src.data.satbird_dataset import SatBirdDataset


def test_datasets_generic_properties(request, tmp_path, sample_csv):
    """This test checks that all datasets implement the basic properties and methods."""
    list_datasets = [ButterflyDataset, SatBirdDataset, HeatGuatemalaDataset]
    use_mock = request.config.getoption("--use-mock")
    if use_mock:
        csv_dir = sample_csv
    else:
        assert False, "Real data not available in test environment."

    for ds_class in list_datasets:
        dataset = ds_class(
            data_dir=csv_dir,
            cache_dir=str(tmp_path),
            modalities={"coords": None},
            use_target_data=True,
            use_aux_data=True,
            seed=0,
            mock=use_mock,
        )

        assert len(dataset) > 0, f"{ds_class.__name__} is empty."
        sample = dataset[0]
        assert "eo" in sample, f"'eo' key missing in sample from {ds_class.__name__}."
        assert (
            "coords" in sample["eo"]
        ), f"'coords' key missing in 'eo' of sample from {ds_class.__name__}."
        assert "target" in sample, f"'target' key missing in sample from {ds_class.__name__}."
        assert "aux" in sample, f"'aux' key missing in sample from {ds_class.__name__}."
        assert hasattr(
            dataset, "num_classes"
        ), f"'num_classes' attribute missing in {ds_class.__name__}."
        assert hasattr(
            dataset, "target_names"
        ), f"'target_names' attribute missing in {ds_class.__name__}."
        assert hasattr(
            dataset, "aux_names"
        ), f"'aux_names' attribute missing in {ds_class.__name__}."
        assert hasattr(dataset, "records"), f"'records' attribute missing in {ds_class.__name__}."
        assert hasattr(
            dataset, "dataset_name"
        ), f"'dataset_name' attribute missing in {ds_class.__name__}."
        assert hasattr(dataset, "mode"), f"'mode' attribute missing in {ds_class.__name__}."
        assert hasattr(
            dataset, "use_features"
        ), f"'use_features' attribute missing in {ds_class.__name__}."
        assert hasattr(
            dataset, "use_aux_data"
        ), f"'use_aux_data' attribute missing in {ds_class.__name__}."
        assert hasattr(
            dataset, "use_target_data"
        ), f"'use_target_data' attribute missing in {ds_class.__name__}."
        assert hasattr(
            dataset, "tabular_dim"
        ), f"'tabular_dim' attribute missing in {ds_class.__name__}."
        assert hasattr(dataset, "setup"), f"'setup' method missing in {ds_class.__name__}."
        assert hasattr(
            dataset, "get_records"
        ), f"'get_records' method missing in {ds_class.__name__}."


def test_datamodule_random_split_and_loaders(create_butterfly_dataset):
    dataset, dm = create_butterfly_dataset

    assert len(dm.data_train) == 4
    assert len(dm.data_val) == 1
    assert len(dm.data_test) == 1

    batch = next(iter(dm.train_dataloader()))
    assert batch["eo"]["coords"].shape == (2, 2)
    assert batch["target"].shape == (2, 2)


def test_random_split_is_deterministic(create_butterfly_dataset):
    dataset1, dm1 = create_butterfly_dataset
    dataset2, dm2 = create_butterfly_dataset

    assert dm1.data_train.indices == dm2.data_train.indices
    assert dm1.data_val.indices == dm2.data_val.indices
    assert dm1.data_test.indices == dm2.data_test.indices
