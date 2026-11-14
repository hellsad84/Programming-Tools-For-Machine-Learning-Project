"""Independent Teammate 3 checks using test-only fake providers."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch
import torch
from week2.train import train_step, save_checkpoint, run as train_run
from week2.evaluate import corpus_bleu, run as evaluate_run
from week2.summarize import summarize

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.generator = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LogSoftmax(-1))
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.embed(tgt)

def batch(indices):
    n = len(indices)
    return NS(src=torch.tensor([[1,4,2]]*n), tgt=torch.tensor([[1,4,2]]*n),
              tgt_y=torch.tensor([[4,2,0]]*n), src_mask=torch.ones(n,1,3,dtype=torch.bool),
              tgt_mask=torch.ones(n,3,3,dtype=torch.bool), ids=[str(i) for i in indices],
              references=['das ist ein test']*n, source_texts=['this is a test']*n)

class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
    def test_update_and_token_count(self):
        model = TinyModel(); opt = torch.optim.Adam(model.parameters())
        before = model.generator[0].weight.detach().clone()
        loss, count = train_step(model, batch([0]), opt, 1.0)
        self.assertEqual(count,2); self.assertGreater(loss,0)
        self.assertFalse(torch.equal(before,model.generator[0].weight))
    def test_empty_targets_rejected(self):
        model = TinyModel(); b = batch([0]); b.tgt_y.zero_()
        with self.assertRaises(ValueError):
            train_step(model,b,torch.optim.Adam(model.parameters()),1.0)
    def test_padding_does_not_affect_update(self):
        a,b = TinyModel(),TinyModel(); b.load_state_dict(a.state_dict())
        short= batch([0]); short.tgt=short.tgt[:,:2]; short.tgt_y=short.tgt_y[:,:2]
        la,_=train_step(a,short,torch.optim.SGD(a.parameters(),lr=.1),1.0)
        lb,_=train_step(b,batch([0]),torch.optim.SGD(b.parameters(),lr=.1),1.0)
        self.assertAlmostEqual(la,lb,places=5)
        for x,y in zip(a.parameters(),b.parameters()): self.assertTrue(torch.allclose(x,y))
    def test_checkpoint_reload(self):
        model=TinyModel()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt'
            save_checkpoint(path,model,torch.optim.Adam(model.parameters()),{'step':1})
            state=torch.load(path,weights_only=True); other=TinyModel()
            other.load_state_dict(state['model_state'])
            for x,y in zip(model.parameters(),other.parameters()): self.assertTrue(torch.equal(x,y))
    def test_bleu_alignment(self):
        with self.assertRaises(ValueError): corpus_bleu(['a'],[])
        score=corpus_bleu(['das ist ein test'],['das ist ein test'])
        self.assertAlmostEqual(score['bleu'],100.0); self.assertIn('tok:13a',score['signature'])
    def test_full_orchestration_with_fake_providers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; data.mkdir()
            (data/'manifest.json').write_text('{"fixture":true}')
            cfg={'data_dir':str(data),'layers':1,'d_model':4,'d_ff':8,'dropout':.1,
                 'batch_size':2,'eval_batch_size':2,'steps':3,'seed':11,'cpu_threads':1,
                 'learning_rate':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.,
                 'clip_norm':1.,'log_every':2,'save_every':2,'max_new_tokens':8}
            config=root/'config.json'; config.write_text(json.dumps(cfg))
            modules={n:ModuleType(n) for n in ['week2.data_pipeline','week2.model','week2.decode']}
            modules['week2.data_pipeline'].load_artifacts=lambda _: (
                NS(get_piece_size=lambda:8),{'files':{'tokenizer.model':'fixture'}})
            class Loader:
                dataset=list(range(5))
                def __init__(self,size): self.size=size
                def __iter__(self):
                    for i in range(0,5,self.size): yield batch(self.dataset[i:i+self.size])
            modules['week2.data_pipeline'].make_loader=lambda d,s,b,se,sh:Loader(b)
            modules['week2.model'].build_model=lambda *a:TinyModel()
            modules['week2.decode'].translate_batch=lambda m,s,mask,t,limit:['das ist ein test']*len(s)
            def evaluation(head,out,limit=None):
                evaluate_run(NS(checkpoint=str(root/f'h{head}/final.pt'),out=str(out),limit=limit,split='valid'))
            with patch.dict('sys.modules',modules), contextlib.redirect_stdout(io.StringIO()):
                for head in [1,8]:
                    target=root/f'h{head}'
                    args=NS(config=str(config),seed=None,steps=None,out=str(target),heads=head,device='cpu')
                    train_run(args); self.assertTrue((target/'step_2.pt').exists())
                    with self.assertRaises(FileExistsError): train_run(args)
                    evaluation(head,target/'eval')
                summarize(root/'h1',root/'h8',root/'summary.md')
                self.assertTrue((root/'summary.md').exists())
                metrics=json.loads((root/'h1/eval/metrics.json').read_text())
                self.assertEqual(metrics['evaluated_pairs'],5); self.assertEqual(metrics['tokens_seen'],10)
                evaluation(1,root/'limited',3)
                self.assertEqual(len((root/'limited/valid.ids.txt').read_text().splitlines()),3)
                (data/'manifest.json').write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'manifest differs'): evaluation(1,root/'bad')
                p=root/'h8/eval/metrics.json'; result=json.loads(p.read_text()); result['steps']=4
                p.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError,'steps'):
                    summarize(root/'h1',root/'h8',root/'invalid.md')

if __name__=='__main__': unittest.main()
