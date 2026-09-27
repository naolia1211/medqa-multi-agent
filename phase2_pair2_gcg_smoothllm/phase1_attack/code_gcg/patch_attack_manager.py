# Patch llm_attacks/base/attack_manager.py -> lop ModelWorker.
# Repo goc nap model 2 ban (tien trinh cha + worker spawn) -> OOM tren 1 GPU 24GB.
# Patch: (1) ModelWorker chay DONG BO trong tien trinh chinh (bo multiprocessing spawn)
#            -> chi 1 ban model tren GPU;
#         (2) model.requires_grad_(False) -> GCG chi can grad theo one-hot dau vao,
#            khong luu grad tham so (tiet kiem ~13GB VRAM).
# Chay: python patch_attack_manager.py   (sau khi da clone repo)
f = "/workspace/llm-attacks/llm_attacks/base/attack_manager.py"
s = open(f).read()

old1 = '''        ).to(device).eval()
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.tasks = mp.JoinableQueue()
        self.results = mp.JoinableQueue()
        self.process = None'''
new1 = '''        ).to(device).eval()
        self.model.requires_grad_(False)   # GCG chi can grad theo one-hot dau vao; bo grad tham so de tiet kiem VRAM
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        import queue as _queue
        self.tasks = _queue.Queue()
        self.results = _queue.Queue()
        self.process = None'''
assert old1 in s, "MISS init"
s = s.replace(old1, new1, 1)

old2 = '''    def start(self):
        self.process = mp.Process(
            target=ModelWorker.run,
            args=(self.model, self.tasks, self.results)
        )
        self.process.start()
        print(f"Started worker {self.process.pid} for model {self.model.name_or_path}")
        return self'''
new2 = '''    def start(self):
        print(f"Started in-process worker for model {self.model.name_or_path}")
        return self'''
assert old2 in s, "MISS start"
s = s.replace(old2, new2, 1)

old3 = '''    def stop(self):
        self.tasks.put(None)
        if self.process is not None:
            self.process.join()
        torch.cuda.empty_cache()
        return self'''
new3 = '''    def stop(self):
        torch.cuda.empty_cache()
        return self'''
assert old3 in s, "MISS stop"
s = s.replace(old3, new3, 1)

old4 = '''    def __call__(self, ob, fn, *args, **kwargs):
        self.tasks.put((deepcopy(ob), fn, args, kwargs))
        return self'''
new4 = '''    def __call__(self, ob, fn, *args, **kwargs):
        if fn == "grad":
            with torch.enable_grad():
                self.results.put(ob.grad(*args, **kwargs))
        else:
            with torch.no_grad():
                if fn == "logits":
                    self.results.put(ob.logits(*args, **kwargs))
                elif fn == "contrast_logits":
                    self.results.put(ob.contrast_logits(*args, **kwargs))
                elif fn == "test":
                    self.results.put(ob.test(*args, **kwargs))
                elif fn == "test_loss":
                    self.results.put(ob.test_loss(*args, **kwargs))
                else:
                    self.results.put(fn(*args, **kwargs))
        return self'''
assert old4 in s, "MISS call"
s = s.replace(old4, new4, 1)

open(f, "w").write(s)
print("PATCH OK")
