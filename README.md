# TorchSnippet

```python
def from_cv2(img, unsqueeze=False):
    img = img[..., ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.tensor(img, dtype=torch.float32)/255.
    if unsqueeze: img = img.unsqueeze(0)
    return img
```

```python
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)
```

```python
def benchmark(model, batch_size=1, input_size=[608, 608], times=100, device='cuda'):
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        
        input = torch.rand(batch_size, 3, input_size[0], input_size[1]).to(device)
        for _ in range(10): model(input) #gpu warmup
            
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        avg_time = 0
        for _ in range(0, times):
            input = torch.rand(batch_size,3,input_size[0],input_size[1]).to(device)
            start.record()
            model(input)
            end.record()
            torch.cuda.synchronize()
            avg_time += start.elapsed_time(end)

        avg_time /= times
        return avg_time
```
