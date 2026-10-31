import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloud_crnn import Charset, CRNN, CachedCRNN
from tib_agent.planner import rule_plan, plan_task, decide_review_after_tools, _endpoint_allowed
from tib_agent.pipeline import OCRPipeline
from tib_agent.settings import Settings

class FakeOCR:
    def recognize(self, path):
        return {'text':'བོད་ཡིག་', 'confidence':None, 'demo_only':False}

class Tests(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(rule_plan('第二张图片不要复核')['review_mode'], 'none')
        self.assertEqual(rule_plan('请处理第2张图片')['target_index'], 2)

    def test_reject_unsafe_endpoint(self):
        self.assertFalse(_endpoint_allowed('http://127.0.0.1:8001/v1/chat/completions'))
        self.assertFalse(_endpoint_allowed('https://evil.com/chat/completions'))
        self.assertTrue(_endpoint_allowed('https://test.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions'))

    def test_fallback(self):
        p = plan_task('只导出JSON，不要复核', Settings())
        self.assertEqual(p['source'], 'rule_fallback')
        self.assertEqual(p['review_mode'], 'none')
        self.assertEqual(p['export_format'], 'json')

    @patch('tib_agent.planner.requests.post')
    def test_real_qwen_plan_parsing(self, post):
        post.return_value = Mock(status_code=200)
        post.return_value.json.return_value = {'choices':[{'message':{'content':
            '{"review_mode":"auto","target_index":null,"export_format":"txt_json"}'}}]}
        params=Settings(enable_qwen=True, qwen_api_url='https://x.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions', qwen_api_key='test')
        out = plan_task('识别所有图片', params)
        self.assertEqual(out['source'], 'qwen_api')

    def test_cached_crnn_loads(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            charset=Charset(['བ','ོ','ད'])
            model=CRNN(charset.num_classes)
            torch.save({'model': model.state_dict(), 'charset':{'chars':charset.chars}, 'args':{'img_height':48,'max_width':1024}}, path/'best.pt')
            Image.fromarray(np.full((50,200,3),255,dtype=np.uint8)).save(path/'line.png')
            ocr=CachedCRNN(path/'best.pt')
            result=ocr.recognize(path/'line.png')
            self.assertEqual(result['confidence'],None)
            self.assertEqual(result['ctc_steps'],48)
            self.assertIsInstance(result['text'],str)

    def test_pipeline_no_api(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            Image.fromarray(np.full((66,426,3),255,dtype=np.uint8)).save(path/'test.png')
            opts=Settings(output_dir=path/'out', enable_qwen=False)
            out = OCRPipeline(settings=opts,ocr=FakeOCR()).process_many([str(path/'test.png')],task='不需要复核')
            self.assertEqual(out['agent_runtime'],'rule_fallback')
            self.assertEqual(out['records'][0]['recognized_text'],'བོད་ཡིག་')
            self.assertTrue(Path(out['archive_path']).exists())

if __name__ == '__main__':
    unittest.main()
