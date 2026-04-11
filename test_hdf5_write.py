"""test_hdf5_write.py — verify h5py vlen image write works."""
import io, h5py, numpy as np
from PIL import Image

# Make a fake JPEG
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buf = io.BytesIO()
img.save(buf, format='JPEG', quality=85)
jpeg_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
print(f"JPEG bytes: {len(jpeg_bytes)}")

# Try writing with vlen_dtype
path = "test_vlen.hdf5"
with h5py.File(path, 'w') as hf:
    imgs = hf.create_group("images")
    try:
        dt = h5py.vlen_dtype(np.dtype('uint8'))
        print("Using vlen_dtype")
    except AttributeError:
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        print("Using special_dtype (legacy)")
    ds = imgs.create_dataset("rgb_left", shape=(3,), dtype=dt)
    ds[0] = jpeg_bytes
    ds[1] = jpeg_bytes
    ds[2] = jpeg_bytes
    print(f"Written: ds.shape={ds.shape}")

# Verify read
with h5py.File(path, 'r') as hf:
    print("Keys:", list(hf["images"].keys()))
    rb = bytes(hf["images"]["rgb_left"][0])
    img2 = Image.open(io.BytesIO(rb))
    print(f"Read back: {img2.size} {img2.mode} — OK")

import os; os.remove(path)
print("\nALL PASS — vlen write works on this machine")
print("Root cause was something else. Check generate_pusht_data.py write_hdf5.")
