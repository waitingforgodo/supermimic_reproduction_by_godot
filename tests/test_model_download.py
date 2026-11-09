from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import zipfile
from unittest.mock import patch
from g1_supermimic import model_download


class ModelDownloadTests(unittest.TestCase):
    def fixture(self,root,bad_member=False):
        archive=root/'bundle.zip'
        payload=b'trusted checkpoint fixture'
        members={'best.pt':payload,'run_manifest.json':b'{}','best.json':b'{}','MODEL_CARD.md':b'card'}
        if bad_member:members['../escape']=b'bad'
        with zipfile.ZipFile(archive,'w') as bundle:
            for name,value in members.items():bundle.writestr(name,value)
        record=dict(url=archive.as_uri(),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                    checkpoint_sha256=hashlib.sha256(payload).hexdigest())
        (root/'models').mkdir()
        (root/'models/manifest.json').write_text(json.dumps(dict(models=dict(skills=record))))
        return record

    def test_valid_bundle_installs_and_existing_different_weights_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fixture(root)
            with patch.object(model_download,'ROOT',root):
                model_download.fetch(['skills'])
                checkpoint=root/'models/skills/best.pt'
                self.assertEqual(checkpoint.read_bytes(),b'trusted checkpoint fixture')
                model_download.fetch(['skills'])
                checkpoint.write_bytes(b'user training')
                with self.assertRaises(FileExistsError):model_download.fetch(['skills'])
                self.assertEqual(checkpoint.read_bytes(),b'user training')

    def test_hash_mismatch_does_not_publish_partial_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            record=self.fixture(root)
            record['archive_sha256']='0'*64
            (root/'models/manifest.json').write_text(json.dumps(dict(models=dict(skills=record))))
            with patch.object(model_download,'ROOT',root),self.assertRaises(ValueError):
                model_download.fetch(['skills'])
            self.assertFalse((root/'models/skills').exists())

    def test_unexpected_and_traversing_archive_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            self.fixture(root,bad_member=True)
            with patch.object(model_download,'ROOT',root),self.assertRaises(ValueError):
                model_download.fetch(['skills'])
            self.assertFalse((root/'escape').exists())
            self.assertFalse((root/'models/skills').exists())
