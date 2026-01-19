import json, os
import pandas as pd
import pytest

from src.data.base_caption_builder import BaseCaptionBuilder, DummyCaptionBuilder
from src.data.butterfly_caption_builder import ButterflyCaptionBuilder
from src.data.base_datamodule import BaseDataModule
from src.data.butterfly_dataset import ButterflyDataset

def test_datamodule_uses_collate_when_aux_data(sample_csv, tmp_path):
    templates_path = tmp_path / "templates.json"
    templates_path.write_text(json.dumps(["<name_loc> text"]))
    caption_builder = DummyCaptionBuilder(str(templates_path), data_dir=str(tmp_path), seed=0)

    dataset = ButterflyDataset(
        path_csv=sample_csv,
        modalities=["coords"],
        use_target_data=True,
        use_aux_data=True,
        seed=0,
    )

    dm = BaseDataModule(
        dataset,
        batch_size=2,
        train_val_test_split=(4, 1, 1),
        split_mode="random",
        caption_builder=caption_builder,
        num_workers=0,
        pin_memory=False,
    )

    batch = next(iter(dm.train_dataloader()))
    assert "text" in batch
    assert len(batch["text"]) == dm.batch_size_per_device

def test_captionbuilder_generic_properties(tmp_path):
    '''This test checks that all caption builders implement the basic properties and methods'''
    dict_caption_builders = {'butterfly': ButterflyCaptionBuilder, 'dummy': DummyCaptionBuilder}

    for name_cb, cb_class in dict_caption_builders.items():
        if name_cb == 'butterfly':
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            templates_path = os.path.join(repo_root, 'data', 'caption_templates', 'butterfly.json')
        else:
            templates_path = tmp_path / "templates.json"
            templates_path.write_text(json.dumps(["<name_loc> text"]))

        caption_builder = cb_class(
            templates_path=str(templates_path),
            data_dir=str(tmp_path),
            seed=0,
        )

        assert hasattr(caption_builder, "templates"), f"'templates' attribute missing in {cb_class.__name__}."
        assert hasattr(caption_builder, "data_dir"), f"'data_dir' attribute missing in {cb_class.__name__}."
        assert hasattr(caption_builder, "seed"), f"'seed' attribute missing in {cb_class.__name__}."
        assert hasattr(caption_builder, "column_to_metadata_map"), f"'column_to_metadata_map' attribute missing in {cb_class.__name__}."
        assert hasattr(caption_builder, "sync_with_dataset"), f"'sync_with_dataset' method missing in {cb_class.__name__}."
        assert callable(getattr(caption_builder, "sync_with_dataset")), f"'sync_with_dataset' is not callable in {cb_class.__name__}."