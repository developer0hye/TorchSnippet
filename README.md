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
