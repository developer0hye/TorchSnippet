# TorchSnippet

```python
def from_cv2(img):
    img = img[..., ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.tensor(img, dtype=torch.float32)/255.
    return img
```
