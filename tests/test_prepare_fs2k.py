import json
from pathlib import Path
import tempfile
import unittest
from scripts.prepare_fs2k import read_pairs,split_train
class FS2KPreparationTest(unittest.TestCase):
    def test_stratified_split(self):
        rows=[{'image_id':f'{s}/{i}','style':s} for s,n in enumerate([357,351,350]) for i in range(n)]
        a,b=split_train(rows)
        self.assertEqual((len(a),len(b)),(898,160))
        self.assertEqual((a,b),split_train(list(reversed(rows))))
        self.assertFalse({r['image_id'] for r in a}&{r['image_id'] for r in b})
    def test_pair_mapping_and_annotation_style(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for path in ['photo/photo1/image0001.JPG','sketch/sketch1/sketch0001.png']:
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
            (root/'anno.json').write_text(json.dumps([{'image_name':'photo1/image0001','style':2}]))
            row=read_pairs(root,'anno.json')[0]
            self.assertEqual(row['style'],2)
            self.assertEqual(row['sketch'],'sketch/sketch1/sketch0001.png')
            (root/'sketch/sketch1/sketch0001.png').unlink()
            with self.assertRaises(ValueError):read_pairs(root,'anno.json')
if __name__=='__main__':unittest.main()
