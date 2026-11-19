"""Small deterministic tests for frozen OOF-only evaluation; no model fitting."""
from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from ntl.evaluation import (SPARSE_SEEDS, generate_masks, validate_masks, strict_metrics,
                            freeze_masks, load_masks)
from ntl.io import digest
from ntl.labels import class_order
from run_s07a import reference_data, MASKS, RESULT, ROOT


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.ref = pd.DataFrame([{'id': f'{p}{i:03}', 'prompt_id': str(p), 'fold': p + 1}
                                 for p in range(2) for i in range(18)])
        self.truth = pd.Series(list(class_order) * 2, index=self.ref.id)

    def test_order_independent_reproducible(self):
        a = generate_masks(self.ref)
        pd.testing.assert_frame_equal(a, generate_masks(self.ref.sample(frac=1, random_state=4)))
        self.assertEqual(a.seed.nunique(), 10)
        self.assertTrue(a.groupby(['seed','prompt_id']).size().eq(2).all())
        self.assertFalse(a.duplicated(['seed','id']).any())
        self.assertTrue(a.id.str.startswith(('0','1')).all())

    def test_label_blind(self):
        a = generate_masks(self.ref)
        altered = self.ref.assign(model='illegal', predicted_model='changed', score=np.nan)
        pd.testing.assert_frame_equal(a, generate_masks(altered))

    def test_small_groups_not_duplicated(self):
        tiny = self.ref.iloc[[0,18]].copy()
        m = generate_masks(tiny)
        self.assertEqual(len(m), 20)
        validate_masks(tiny,m)

    def test_reference_errors(self):
        for bad in [pd.concat([self.ref,self.ref.iloc[[0]]]), self.ref.assign(fold=0),
                    self.ref.assign(fold=1.0), self.ref.assign(id=12),
                    self.ref.assign(prompt_id='same')]:
            with self.assertRaises(ValueError): generate_masks(bad)
        with self.assertRaises(ValueError): generate_masks(self.ref,seeds=(1,1))

    def test_mask_corruption(self):
        good = generate_masks(self.ref)
        variants = [good.iloc[1:],pd.concat([good,good.iloc[[0]]]),good.assign(fold=0),
                    good.assign(fold=1.0),good.assign(seed=0),good.assign(prompt_id='wrong'),
                    good.assign(id='extra')]
        for bad in variants:
            with self.assertRaises(ValueError): validate_masks(self.ref,bad)

    def test_strict_official_metrics(self):
        m = strict_metrics(class_order,class_order)
        self.assertEqual(m['balanced_accuracy'],1)
        self.assertEqual(m['macro_f1'],1)
        self.assertEqual(m['n_classes'],18)
        m = strict_metrics(class_order,[class_order[0]]*18)
        self.assertAlmostEqual(m['balanced_accuracy'],1/18)
        self.assertAlmostEqual(m['accuracy'],1/18)

    def test_absent_class_not_silently_dropped(self):
        m = strict_metrics(class_order[:2],class_order[:2])
        self.assertIsNone(m['balanced_accuracy'])
        self.assertEqual(len(m['missing_classes']),16)
        self.assertIsNone(m['recall'][class_order[-1]])
        self.assertAlmostEqual(m['macro_f1'],2/18)
        with self.assertRaises(ValueError): strict_metrics(['unknown'],['unknown'])
        with self.assertRaises(ValueError): strict_metrics([],[])

    def test_freeze_no_overwrite_and_bad_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); train=tmp/'train.csv'; fold=tmp/'fold.csv'
            train.write_text('original',encoding='utf-8'); fold.write_text('original fold',encoding='utf-8')
            folder=tmp/'resource'
            freeze_masks(self.ref,self.truth,train,fold,folder)
            load_masks(self.ref,train,fold,folder)
            with self.assertRaises(FileExistsError): freeze_masks(self.ref,self.truth,train,fold,folder)
            for target in [train,fold,folder/'masks.csv']:
                original=target.read_bytes(); target.write_bytes(original+b'changed')
                with self.assertRaises(ValueError): load_masks(self.ref,train,fold,folder)
                target.write_bytes(original)

    def test_official_masks_and_training_boundary(self):
        train,ref,path,fold=reference_data()
        masks,meta=load_masks(ref,path,fold,MASKS)
        pd.testing.assert_frame_equal(generate_masks(ref),generate_masks(ref.sample(frac=1,random_state=91)))
        for seed in SPARSE_SEEDS:
            one=masks[masks.seed==seed]
            self.assertEqual(len(one),300)
            for f in range(1,6):
                held=ref[ref.fold==f]; fit=ref[ref.fold!=f]
                selected=one[one.fold==f]
                self.assertTrue(set(selected.id).issubset(set(held.id)))
                self.assertFalse(set(fit.prompt_id)&set(held.prompt_id))
                self.assertFalse((set(held.id)-set(selected.id))&set(fit.id))

    def test_real_results_same_masks_and_scores(self):
        meta=json.loads((RESULT/'evaluation_metadata.json').read_text(encoding='utf-8'))
        use=pd.read_csv(RESULT/'mask_use.csv')
        self.assertEqual(meta['new_model_fits'],0)
        self.assertEqual(use.mask_sha256.nunique(),1)
        self.assertEqual(len(use),30)
        from run_s07a import RUNS
        train,ref,path,fold=reference_data(); masks,_=load_masks(ref,path,fold,MASKS)
        saved=pd.read_csv(RESULT/'sparse_metrics.csv')
        for name,run in RUNS.items():
            frame=pd.read_parquet(ROOT/'artifacts'/run/'oof_predictions.parquet').set_index('id')
            for seed in SPARSE_SEEDS:
                one=frame.loc[masks.loc[masks.seed==seed,'id']]
                actual=strict_metrics(one.true_model,one.predicted_model)
                row=saved[(saved.model==name)&(saved.seed==seed)].iloc[0]
                for key in ('balanced_accuracy','accuracy','macro_f1'):
                    self.assertAlmostEqual(actual[key],row[key],places=14)

    def test_existing_resources_and_v1_history_unchanged(self):
        before=json.loads((ROOT/'artifacts/s07a_protocol_20261119_01/before.json').read_text(encoding='utf-8'))
        for name,expected in before['protected_sha256'].items():
            self.assertEqual(digest(ROOT/name),expected,name)
        self.assertEqual(digest(ROOT/'docs/history/工作计划_v1.1_20261119.md'),before['plan_v1_1_sha256'])

    def test_frozen_coverage_metadata(self):
        train,ref,path,fold=reference_data(); masks,meta=load_masks(ref,path,fold,MASKS)
        labels=train.set_index('id').model
        for row in meta['support']:
            one=masks[masks.seed==row['seed']]
            if row['fold']!='aggregate': one=one[one.fold==row['fold']]
            expected={label:int(labels.loc[one.id].eq(label).sum()) for label in class_order}
            self.assertEqual(row['support'],expected)

    def test_evaluation_sources_and_receipt(self):
        meta=json.loads((RESULT/'evaluation_metadata.json').read_text(encoding='utf-8'))
        for path,expected in meta['source_sha256'].items():
            self.assertEqual(digest(ROOT/path),expected,path)
            self.assertEqual(digest(RESULT/'source'/path),expected,path)
        receipt=json.loads((MASKS/'freeze-receipt.json').read_text(encoding='utf-8'))
        self.assertEqual(digest(MASKS/'masks.csv'),receipt['mask_sha256'])
        self.assertEqual(digest(MASKS/'metadata.json'),receipt['metadata_sha256'])
        self.assertEqual(digest(ROOT/'experiments/folds/folds_v1_metadata.json'),receipt['fold_metadata_sha256'])
        self.assertEqual(digest(ROOT/'experiments/folds/folds_v2.csv'),receipt['folds_v2_sha256'])
        self.assertEqual(digest(ROOT/'experiments/folds/folds_v2_metadata.json'),receipt['folds_v2_metadata_sha256'])

    def test_document_links_and_plan_mapping(self):
        import re
        from urllib.parse import unquote
        files=[ROOT/'docs/工作计划.md',ROOT/'docs/实验评价协议.md',ROOT/'docs/S07-A_研究资料核实.md',
               ROOT/'docs/history/README.md',ROOT/'README.md',RESULT/'report.md']
        for path in files:
            text=path.read_text(encoding='utf-8')
            for link in re.findall(r'\]\(([^)]+)\)',text):
                if '://' in link or link.startswith('#'): continue
                target=unquote(link.split('#')[0]).strip('<>')
                self.assertTrue((path.parent/target).exists(),f'{path.name}: broken {link}')
        plan=files[0].read_text(encoding='utf-8')
        self.assertIn('计划版本：v2.0',plan)
        for marker in ['S07-A','S07-B','S08-A','S08-B','S09-L01','S10-01','S11-01','旧S09-01','B0','B1','B2']:
            self.assertIn(marker,plan)


if __name__=='__main__':
    unittest.main(verbosity=2)
