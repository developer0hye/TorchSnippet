# TorchSnippet

```python
def from_cv2(img, unsqueeze=False):
    img = img[..., ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.tensor(img, dtype=torch.float32)/255.
    if unsqueeze: img = img.unsqueeze(0)
    return img
```
